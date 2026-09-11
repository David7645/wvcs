import json, math, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score

YEARS=range(2015,2027)
BASE="https://raw.githubusercontent.com/FinanceData/marcap/master/data/marcap-%d.parquet"

def load():
    frames=[]
    for y in YEARS:
        print("LOAD", y, flush=True)
        df=pd.read_parquet(BASE%y)
        if "Date" not in df.columns:
            raise RuntimeError("Date column missing")
        df["Date"]=pd.to_datetime(df["Date"])
        frames.append(df)
    d=pd.concat(frames,ignore_index=True)
    d=d[d["Market"].astype(str).str.upper().eq("KOSDAQ")].copy()
    for c in ["Open","High","Low","Close","Volume","Amount","Marcap","ChangesRatio"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["Code","Date","Open","High","Low","Close","Volume","Amount","Marcap"])
    d=d[(d.Open>0)&(d.High>0)&(d.Low>0)&(d.Close>0)]
    d["Code"]=d["Code"].astype(str).str.zfill(6)
    d["ChangeCode"]=d["ChangeCode"].astype(str).str.replace(".0","",regex=False)
    return d.sort_values(["Code","Date"]).reset_index(drop=True)

def make_features(d):
    # market-day features available by D0 close
    daily=d.groupby("Date").agg(
        breadth=("ChangesRatio",lambda s: float((s>0).mean())),
        mkt_med_ret=("ChangesRatio","median"),
        limitup_count=("ChangeCode",lambda s:int((s.astype(str)=="1").sum())),
        mkt_amount=("Amount","sum"),
    )
    d=d.join(daily,on="Date")
    out=[]
    for code,g in d.groupby("Code",sort=False):
        g=g.sort_values("Date").copy()
        c=g["Close"]; v=g["Volume"]; a=g["Amount"]
        g["ret5"]=c.pct_change(5)*100
        g["ret20"]=c.pct_change(20)*100
        g["ret60"]=c.pct_change(60)*100
        g["accel"]=g["ret5"]-(c.pct_change(10)*100)/2.0
        g["rv20"]=c.pct_change().rolling(20).std()*100
        g["vol_ratio20"]=v/(v.shift(1).rolling(20).median().replace(0,np.nan))
        g["amt_ratio20"]=a/(a.shift(1).rolling(20).median().replace(0,np.nan))
        g["high20_dist"]=c/(g["High"].shift(1).rolling(20).max())-1
        g["high60_dist"]=c/(g["High"].shift(1).rolling(60).max())-1
        g["close_loc"]=(c-g["Low"])/(g["High"]-g["Low"]).replace(0,np.nan)
        g["turnover_marcap"]=g["Amount"]/g["Marcap"].replace(0,np.nan)
        g["prior_lu60"]=(g["ChangeCode"].shift(1).eq("1").rolling(60).sum())
        g["marcap_log"]=np.log1p(g["Marcap"])
        g["amount_log"]=np.log1p(g["Amount"])
        g["gap0"]=(g["Open"]/g["Close"].shift(1)-1)*100
        # exact official upper-limit code; fallback only if historical code absent on row
        is_lu=g["ChangeCode"].eq("1")
        fallback=(g["ChangesRatio"]>=29.5)&(np.isclose(g["Close"],g["High"]))
        event_idx=np.where((is_lu|fallback).values)[0]
        for i in event_idx:
            if i<60 or i+60>=len(g):
                continue
            r=g.iloc[i]
            featcols=["close_loc","ret5","ret20","ret60","accel","rv20","vol_ratio20","amt_ratio20",
                      "high20_dist","high60_dist","turnover_marcap","prior_lu60","marcap_log","amount_log",
                      "breadth","mkt_med_ret","limitup_count","gap0"]
            if any(pd.isna(r[x]) or np.isinf(r[x]) for x in featcols):
                continue
            fut=g.iloc[i+1:i+61].reset_index(drop=True)
            if len(fut)<10: continue
            d1=fut.iloc[0]
            d1_gap=(d1.Open/r.Close-1)*100
            # path types
            type1=bool(str(d1.ChangeCode)=="1" or (d1.ChangesRatio>=29.5 and np.isclose(d1.Close,d1.High)))
            first10=fut.iloc[:10]
            dd10=float(first10.Low.min()/r.Close-1)*100
            reup10=bool((first10.High>=r.Close*1.15).any() or (first10.ChangeCode.astype(str)=="1").any())
            type2=(not type1) and dd10>=-10.0 and reup10
            # failure if -10% close is reached before any +15% high/re-limit-up
            type4=False
            seen_reup=False
            for _,q in first10.iterrows():
                if q.High>=r.Close*1.15 or str(q.ChangeCode)=="1":
                    seen_reup=True; break
                if q.Close<=r.Close*0.90:
                    type4=True; break
            if type1: path_type=1
            elif type2: path_type=2
            elif type4: path_type=4
            else:
                long=fut.iloc[10:60]
                if len(long):
                    up=long.High.max()/r.Close-1
                    dn=long.Low.min()/r.Close-1
                    path_type=31 if up>=0.15 else (32 if dn<=-0.15 else 30)
                else: path_type=30

            tradable=(-5.0<=d1_gap<=3.0)
            tp_success=False; stopped=False; d5ret=np.nan; mfe=np.nan; mae=np.nan
            if tradable:
                entry=float(d1.Open); horizon=fut.iloc[:5]
                mfe=float(horizon.High.max()/entry-1)*100
                mae=float(horizon.Low.min()/entry-1)*100
                for _,q in horizon.iterrows():
                    if q.High>=entry*1.06:
                        tp_success=True; break
                    if q.Close<=entry*0.97:
                        stopped=True; break
                d5ret=float(horizon.iloc[-1].Close/entry-1)*100 if len(horizon)>=5 else np.nan
            continuation=path_type in (1,2)
            positive=bool(tradable and continuation and tp_success)
            row={"Code":code,"Date":r.Date,"year":int(r.Date.year),"d1_gap":float(d1_gap),
                 "path_type":path_type,"continuation":int(continuation),"failure":int(path_type==4),
                 "tradable":int(tradable),"tp_success":int(tp_success),"positive":int(positive),
                 "mfe":mfe,"mae":mae,"d5ret":d5ret}
            row.update({x:float(r[x]) for x in featcols})
            out.append(row)
    return pd.DataFrame(out)

def wilson_lb(k,n,z=1.6448536269514722): # one-sided 95%
    if n==0:return 0.0
    p=k/n
    den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def fit_prob(X,y,kind):
    if kind=="logit":
        m=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=0.35,max_iter=3000,class_weight="balanced"))])
    else:
        m=HistGradientBoostingClassifier(max_depth=3,learning_rate=.04,max_iter=160,l2_regularization=2.0,min_samples_leaf=18,random_state=42)
    m.fit(X,y)
    return m

d=load()
ev=make_features(d)
ev.to_csv("events_summary.csv",index=False)
print("EVENTS",len(ev),ev.year.value_counts().sort_index().to_dict(),flush=True)

features=["close_loc","ret5","ret20","ret60","accel","rv20","vol_ratio20","amt_ratio20",
          "high20_dist","high60_dist","turnover_marcap","prior_lu60","marcap_log","amount_log",
          "breadth","mkt_med_ret","limitup_count","gap0","d1_gap"]

train=ev[(ev.year>=2015)&(ev.year<=2018)&(ev.tradable==1)].copy()
cal=ev[(ev.year==2019)&(ev.tradable==1)].copy()
test=ev[(ev.year>=2020)&(ev.year<=2026)&(ev.tradable==1)].copy()
if min(len(train),len(cal),len(test))==0: raise RuntimeError("empty split")

# Three statistically distinct heads, each fixed from 2015-2018.
Xtr=train[features]
cont_log=fit_prob(Xtr,train.continuation,"logit")
cont_gb=fit_prob(Xtr,train.continuation,"gb")
fail_log=fit_prob(Xtr,train.failure,"logit")
tp_gb=fit_prob(Xtr,train.tp_success,"gb")

def raw_score(frame):
    X=frame[features]
    pc=.55*cont_log.predict_proba(X)[:,1]+.45*cont_gb.predict_proba(X)[:,1]
    pf=fail_log.predict_proba(X)[:,1]
    pt=tp_gb.predict_proba(X)[:,1]
    # competing-risk score
    return np.clip(pc*(1-pf)*np.sqrt(np.clip(pt,1e-6,1)),1e-6,1-1e-6)

cal=cal.copy(); cal["raw"]=raw_score(cal)
# Platt calibration learned only on 2019.
platt=LogisticRegression(C=1.0,max_iter=2000)
platt.fit(cal[["raw"]],cal["positive"])
cal["p"]=platt.predict_proba(cal[["raw"]])[:,1]

# Threshold freeze using 2019 only: prioritize precision/Wilson LB, require >=12 signals;
# no 2020+ data is touched here.
cands=[]
for t in np.unique(np.round(cal.p,4)):
    sel=cal[cal.p>=t]
    n=len(sel)
    if n<12: continue
    k=int(sel.positive.sum()); prec=k/n; lb=wilson_lb(k,n)
    cands.append((lb,prec,n,float(t)))
if not cands: raise RuntimeError("calibration has fewer than 12 selectable signals")
# choose highest lower bound, then precision, then more n
cands=sorted(cands,key=lambda x:(x[0],x[1],x[2]),reverse=True)
lb,calprec,caln,threshold=cands[0]

test=test.copy(); test["raw"]=raw_score(test); test["p"]=platt.predict_proba(test[["raw"]])[:,1]
test["signal"]=(test.p>=threshold).astype(int)
sig=test[test.signal==1].copy()

yearly=[]
for y in range(2020,2027):
    s=sig[sig.year==y]; n=len(s); k=int(s.positive.sum())
    yearly.append({"year":y,"signals":n,"success":k,"hit_rate":round(100*k/n,2) if n else None,
                   "avg_mfe":round(float(s.mfe.mean()),3) if n else None,
                   "avg_mae":round(float(s.mae.mean()),3) if n else None})

N=len(sig); K=int(sig.positive.sum()); hit=100*K/N if N else 0
ptype={str(k):int(v) for k,v in sig.path_type.value_counts().to_dict().items()}
result={
 "model":"KQ-LUCR v1.0",
 "data_source":"FinanceData/marcap (KRX daily market-cap dataset)",
 "train_period":"2015-2018",
 "calibration_period":"2019",
 "test_period":"2020-2026",
 "event_count_all":int(len(ev)),
 "train_tradable":int(len(train)),
 "calibration_tradable":int(len(cal)),
 "test_tradable":int(len(test)),
 "frozen_threshold":float(threshold),
 "calibration":{"signals":int(caln),"precision_pct":round(calprec*100,2),"wilson_lb_pct":round(lb*100,2)},
 "test":{"signals":N,"success":K,"hit_rate_pct":round(hit,2),"wilson_lb_pct":round(wilson_lb(K,N)*100,2),
         "avg_mfe_pct":round(float(sig.mfe.mean()),3) if N else None,
         "avg_mae_pct":round(float(sig.mae.mean()),3) if N else None,
         "avg_d5_return_pct":round(float(sig.d5ret.mean()),3) if N else None,
         "path_type_counts":ptype},
 "yearly":yearly,
 "pass_80pct":bool(N>=100 and hit>=80.0 and all((x["signals"]==0 or x["hit_rate"]>=70.0) for x in yearly)),
 "criteria":{"overall_hit_rate_min":80.0,"min_signals":100,"yearly_min_when_signal_exists":70.0}
}
with open("result.json","w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
sig[["Code","Date","year","d1_gap","path_type","positive","mfe","mae","d5ret","p"]].to_csv("signals_2020_2026.csv",index=False)
print("RESULT_JSON")
print(json.dumps(result,ensure_ascii=False,indent=2))
