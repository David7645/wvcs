import json, math, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

COLS=["Date","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","ChangeCode","Marcap","Market"]
frames=[]
for y in range(2015,2027):
    p=f"data/marcap-{y}.parquet"
    print("READ",y,flush=True)
    x=pd.read_parquet(p,columns=COLS)
    x=x[x["Market"].astype(str).str.upper().eq("KOSDAQ")].copy()
    frames.append(x)
d=pd.concat(frames,ignore_index=True)
d["Date"]=pd.to_datetime(d["Date"])
d["Code"]=d["Code"].astype(str).str.zfill(6)
d["ChangeCode"]=d["ChangeCode"].astype(str).str.replace(".0","",regex=False)
for c in ["Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap"]:
    d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna(subset=["Date","Code","Open","High","Low","Close","Volume","Amount","Marcap"])
d=d[(d.Open>0)&(d.High>0)&(d.Low>0)&(d.Close>0)].sort_values(["Code","Date"]).reset_index(drop=True)
print("ROWS",len(d),flush=True)

# market features
md=d.groupby("Date").agg(
    breadth=("ChangesRatio",lambda s: (s>0).mean()),
    mkt_med_ret=("ChangesRatio","median"),
    limitup_count=("ChangeCode",lambda s:(s=="1").sum())
)
d=d.join(md,on="Date")

# vectorized security-history features
g=d.groupby("Code",sort=False)
d["prev_close"]=g["Close"].shift(1)
d["ret5"]=(d["Close"]/g["Close"].shift(5)-1)*100
d["ret10"]=(d["Close"]/g["Close"].shift(10)-1)*100
d["ret20"]=(d["Close"]/g["Close"].shift(20)-1)*100
d["ret60"]=(d["Close"]/g["Close"].shift(60)-1)*100
d["accel"]=d["ret5"]-d["ret10"]/2.0
d["gap0"]=(d["Open"]/d["prev_close"]-1)*100
d["close_loc"]=(d["Close"]-d["Low"])/(d["High"]-d["Low"]).replace(0,np.nan)
d["marcap_log"]=np.log1p(d["Marcap"])
d["amount_log"]=np.log1p(d["Amount"])
d["turnover_marcap"]=d["Amount"]/d["Marcap"].replace(0,np.nan)
# group rolling transforms (prior history only)
d["vol_med20"]=g["Volume"].transform(lambda s:s.shift(1).rolling(20,min_periods=15).median())
d["amt_med20"]=g["Amount"].transform(lambda s:s.shift(1).rolling(20,min_periods=15).median())
d["high20_prev"]=g["High"].transform(lambda s:s.shift(1).rolling(20,min_periods=15).max())
d["ret1"]=g["Close"].pct_change()*100
d["rv20"]=d.groupby("Code")["ret1"].transform(lambda s:s.shift(1).rolling(20,min_periods=15).std())
d["vol_ratio20"]=d["Volume"]/d["vol_med20"].replace(0,np.nan)
d["amt_ratio20"]=d["Amount"]/d["amt_med20"].replace(0,np.nan)
d["high20_dist"]=d["Close"]/d["high20_prev"]-1
d["prior_lu60"]=d.groupby("Code")["ChangeCode"].transform(lambda s:(s=="1").shift(1).rolling(60,min_periods=30).sum())

features=["close_loc","ret5","ret20","ret60","accel","rv20","vol_ratio20","amt_ratio20",
          "high20_dist","turnover_marcap","prior_lu60","marcap_log","amount_log",
          "breadth","mkt_med_ret","limitup_count","gap0","d1_gap"]

events=[]
for code,gg in d.groupby("Code",sort=False):
    gg=gg.reset_index(drop=True)
    inds=np.flatnonzero(gg["ChangeCode"].eq("1").values)
    last=-999
    for i in inds:
        # episode dedupe: use only first upper-limit event in any 5-trading-day cluster
        if i-last<=4:
            continue
        last=i
        if i<60 or i+60>=len(gg):
            continue
        r=gg.iloc[i]
        basefeat=["close_loc","ret5","ret20","ret60","accel","rv20","vol_ratio20","amt_ratio20",
                  "high20_dist","turnover_marcap","prior_lu60","marcap_log","amount_log",
                  "breadth","mkt_med_ret","limitup_count","gap0"]
        vals=[r[c] for c in basefeat]
        if any(pd.isna(v) or np.isinf(v) for v in vals):
            continue
        fut=gg.iloc[i+1:i+61].reset_index(drop=True)
        d1=fut.iloc[0]
        d1_gap=(float(d1.Open)/float(r.Close)-1)*100
        first10=fut.iloc[:10]
        type1=(str(d1.ChangeCode)=="1")
        dd10=(first10.Low.min()/r.Close-1)*100
        reup10=bool((first10.High>=r.Close*1.15).any() or first10.ChangeCode.eq("1").any())
        type2=(not type1) and dd10>=-10.0 and reup10
        type4=False
        for _,q in first10.iterrows():
            if q.High>=r.Close*1.15 or str(q.ChangeCode)=="1":
                break
            if q.Close<=r.Close*0.90:
                type4=True
                break
        if type1: ptype=1
        elif type2: ptype=2
        elif type4: ptype=4
        else:
            long=fut.iloc[10:60]
            up=(long.High.max()/r.Close-1)*100
            dn=(long.Low.min()/r.Close-1)*100
            ptype=31 if up>=15 else (32 if dn<=-15 else 30)
        tradable=(-5.0<=d1_gap<=3.0)
        tp=False; mfe=np.nan; mae=np.nan; d5ret=np.nan
        if tradable:
            entry=float(d1.Open)
            hz=fut.iloc[:5]
            mfe=(hz.High.max()/entry-1)*100
            mae=(hz.Low.min()/entry-1)*100
            for _,q in hz.iterrows():
                hit_tp=q.High>=entry*1.06
                hit_stop=q.Close<=entry*0.97
                if hit_tp and hit_stop:
                    tp=False
                    break
                if hit_stop:
                    tp=False
                    break
                if hit_tp:
                    tp=True
                    break
            if len(hz)>=5:
                d5ret=(hz.iloc[-1].Close/entry-1)*100
        cont=ptype in (1,2)
        positive=bool(tradable and cont and tp)
        rec={"Code":code,"Name":str(r.Name),"Date":r.Date,"year":int(r.Date.year),"d1_gap":float(d1_gap),
             "path_type":ptype,"continuation":int(cont),"failure":int(ptype==4),"tradable":int(tradable),
             "tp_success":int(tp),"positive":int(positive),"mfe":mfe,"mae":mae,"d5ret":d5ret}
        rec.update({c:float(r[c]) for c in basefeat})
        events.append(rec)
ev=pd.DataFrame(events)
print("EVENTS",len(ev),ev.year.value_counts().sort_index().to_dict(),flush=True)
ev.to_csv("events_summary.csv",index=False)

def lb(k,n,z=1.6448536269514722):
    if n<=0:return 0.0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def mdl(X,y,kind):
    if kind=="logit":
        m=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.35,max_iter=3000,class_weight="balanced"))])
    else:
        m=HistGradientBoostingClassifier(max_depth=3,learning_rate=.04,max_iter=140,l2_regularization=2.0,min_samples_leaf=15,random_state=42)
    m.fit(X,y); return m

train=ev[(ev.year<=2018)&(ev.tradable==1)].copy()
cal=ev[(ev.year==2019)&(ev.tradable==1)].copy()
test=ev[(ev.year>=2020)&(ev.year<=2026)&(ev.tradable==1)].copy()
print("SPLITS",len(train),len(cal),len(test),flush=True)
X=train[features]
mc1=mdl(X,train.continuation,"logit")
mc2=mdl(X,train.continuation,"gb")
mf=mdl(X,train.failure,"logit")
mt=mdl(X,train.tp_success,"gb")
def raw(z):
    X=z[features]
    pc=.55*mc1.predict_proba(X)[:,1]+.45*mc2.predict_proba(X)[:,1]
    pf=mf.predict_proba(X)[:,1]
    pt=mt.predict_proba(X)[:,1]
    return np.clip(pc*(1-pf)*np.sqrt(np.clip(pt,1e-6,1)),1e-6,1-1e-6)

cal["raw"]=raw(cal)
pl=LogisticRegression(C=1,max_iter=2000).fit(cal[["raw"]],cal.positive)
cal["p"]=pl.predict_proba(cal[["raw"]])[:,1]
# Freeze threshold using 2019 only. Require minimum 12 signals to prevent tiny-N cherry picking.
opts=[]
for t in sorted(set(np.round(cal.p,4))):
    s=cal[cal.p>=t]; n=len(s)
    if n<12:continue
    k=int(s.positive.sum())
    opts.append((lb(k,n),k/n,n,float(t)))
if not opts: raise RuntimeError("2019 calibration produced <12 selectable cases")
wlb,cprec,cn,thr=max(opts,key=lambda q:(q[0],q[1],q[2]))
test["raw"]=raw(test); test["p"]=pl.predict_proba(test[["raw"]])[:,1]
sig=test[test.p>=thr].copy()
N=len(sig);K=int(sig.positive.sum()); hit=100*K/N if N else 0
yr=[]
for y in range(2020,2027):
    s=sig[sig.year==y]; n=len(s); k=int(s.positive.sum())
    yr.append({"year":y,"signals":n,"success":k,"hit_rate_pct":round(100*k/n,2) if n else None,
               "avg_mfe_pct":round(float(s.mfe.mean()),2) if n else None,
               "avg_mae_pct":round(float(s.mae.mean()),2) if n else None,
               "avg_d5_return_pct":round(float(s.d5ret.mean()),2) if n else None})
path_counts={str(k):int(v) for k,v in sig.path_type.value_counts().to_dict().items()}
result={"model":"KQ-LUCR v1.0 FIXED","data":"FinanceData/marcap KOSDAQ official ChangeCode upper limits",
"train":"2015-2018","calibration":"2019","test":"2020-2026","episode_dedupe":"first upper-limit in 5-trading-day cluster",
"entry":"D+1 open only if gap -5% to +3%","success":"path 1/2 AND +6% TP before -3% close-stop within D+5; same-day TP+stop counted fail",
"events_total":int(len(ev)),"splits":{"train":len(train),"calibration":len(cal),"test":len(test)},
"threshold":thr,"calibration_result":{"signals":cn,"precision_pct":round(cprec*100,2),"one_sided_95_wilson_lb_pct":round(wlb*100,2)},
"test_result":{"signals":N,"success":K,"hit_rate_pct":round(hit,2),"one_sided_95_wilson_lb_pct":round(lb(K,N)*100,2),
"avg_mfe_pct":round(float(sig.mfe.mean()),2) if N else None,"avg_mae_pct":round(float(sig.mae.mean()),2) if N else None,
"avg_d5_return_pct":round(float(sig.d5ret.mean()),2) if N else None,"path_type_counts":path_counts},
"yearly":yr,
"pass_80":bool(N>=100 and hit>=80 and all(x["signals"]==0 or x["hit_rate_pct"]>=70 for x in yr))}
open("result.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
sig[["Code","Name","Date","year","d1_gap","path_type","positive","mfe","mae","d5ret","p"]].to_csv("signals_2020_2026.csv",index=False)
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
