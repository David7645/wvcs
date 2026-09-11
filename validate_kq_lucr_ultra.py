import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

YEARS = range(2015, 2027)
COLS = ["Date","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","ChangeCode","Marcap","Market"]

frames=[]
for y in YEARS:
    print("READ", y, flush=True)
    p=f"data/marcap-{y}.parquet"
    try:
        x=pd.read_parquet(p, columns=COLS, filters=[("Market","=","KOSDAQ")])
    except Exception:
        x=pd.read_parquet(p, columns=COLS)
        x=x[x["Market"].astype(str).str.upper().eq("KOSDAQ")]
    frames.append(x)
d=pd.concat(frames, ignore_index=True)
d["Date"]=pd.to_datetime(d["Date"])
d["Code"]=d["Code"].astype(str).str.zfill(6)
d["ChangeCode"]=d["ChangeCode"].astype(str).str.replace(".0","",regex=False)
for c in ["Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap"]:
    d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna(subset=["Date","Code","Open","High","Low","Close","Volume","Amount","Marcap"])
d=d[(d.Open>0)&(d.High>0)&(d.Low>0)&(d.Close>0)].copy()

# D0-close market context, available without look-ahead.
breadth=d.groupby("Date",sort=False)["ChangesRatio"].apply(lambda s: float((s>0).mean())).to_dict()
medret=d.groupby("Date",sort=False)["ChangesRatio"].median().to_dict()
lucount=d[d["ChangeCode"].eq("1")].groupby("Date").size().to_dict()

# Only securities that ever produced an official KRX upper-limit code are needed for event extraction.
event_codes=set(d.loc[d["ChangeCode"].eq("1"),"Code"])
d=d[d["Code"].isin(event_codes)].sort_values(["Code","Date"]).reset_index(drop=True)
print("KOSDAQ EVENT-CODE ROWS",len(d),"EVENT CODES",len(event_codes),flush=True)

base_features=["close_loc","ret5","ret20","ret60","accel","rv20","vol_ratio20","amt_ratio20",
               "high20_dist","turnover_marcap","prior_lu60","marcap_log","amount_log",
               "breadth","mkt_med_ret","limitup_count","gap0"]
events=[]

for code, gg in d.groupby("Code", sort=False):
    gg=gg.reset_index(drop=True)
    cc=gg["ChangeCode"].astype(str).to_numpy()
    inds=np.flatnonzero(cc=="1")
    if len(inds)==0:
        continue

    op=gg["Open"].to_numpy(float); hi=gg["High"].to_numpy(float); lo=gg["Low"].to_numpy(float)
    cl=gg["Close"].to_numpy(float); vo=gg["Volume"].to_numpy(float); am=gg["Amount"].to_numpy(float)
    mc=gg["Marcap"].to_numpy(float); dt=gg["Date"].to_numpy()
    nm=gg["Name"].astype(str).to_numpy()
    chg=gg["ChangesRatio"].to_numpy(float)

    last_event=-999
    for i in inds:
        # One independent episode: only first limit-up inside a 5-trading-day cluster.
        if i-last_event<=4:
            continue
        last_event=i
        if i<60 or i+60>=len(gg):
            continue

        if cl[i-5]<=0 or cl[i-10]<=0 or cl[i-20]<=0 or cl[i-60]<=0:
            continue

        ret5=(cl[i]/cl[i-5]-1)*100
        ret10=(cl[i]/cl[i-10]-1)*100
        ret20=(cl[i]/cl[i-20]-1)*100
        ret60=(cl[i]/cl[i-60]-1)*100
        accel=ret5-ret10/2.0

        # Exclude D0 from baseline windows to avoid mixing event shock into baseline.
        prev20_cl=cl[i-20:i+1]
        prev20_ret=(prev20_cl[1:]/prev20_cl[:-1]-1)*100
        rv20=float(np.nanstd(prev20_ret, ddof=1)) if len(prev20_ret)>1 else np.nan
        vol_base=float(np.nanmedian(vo[i-20:i]))
        amt_base=float(np.nanmedian(am[i-20:i]))
        prev_high=float(np.nanmax(hi[i-20:i]))
        vol_ratio=vo[i]/vol_base if vol_base>0 else np.nan
        amt_ratio=am[i]/amt_base if amt_base>0 else np.nan
        high20_dist=cl[i]/prev_high-1 if prev_high>0 else np.nan
        prior_lu60=int(np.sum(cc[i-60:i]=="1"))
        close_loc=(cl[i]-lo[i])/(hi[i]-lo[i]) if hi[i]>lo[i] else 1.0
        turnover=am[i]/mc[i] if mc[i]>0 else np.nan
        gap0=(op[i]/cl[i-1]-1)*100 if cl[i-1]>0 else np.nan
        date=pd.Timestamp(dt[i])

        feature_vals={
            "close_loc":close_loc,"ret5":ret5,"ret20":ret20,"ret60":ret60,"accel":accel,
            "rv20":rv20,"vol_ratio20":vol_ratio,"amt_ratio20":amt_ratio,
            "high20_dist":high20_dist,"turnover_marcap":turnover,"prior_lu60":prior_lu60,
            "marcap_log":math.log1p(mc[i]),"amount_log":math.log1p(am[i]),
            "breadth":float(breadth.get(date,np.nan)),"mkt_med_ret":float(medret.get(date,np.nan)),
            "limitup_count":float(lucount.get(date,0)),"gap0":gap0,
        }
        if any((not np.isfinite(v)) for v in feature_vals.values()):
            continue

        d1_gap=(op[i+1]/cl[i]-1)*100
        # 1: next-day consecutive upper limit.
        type1=(cc[i+1]=="1")
        # 2: no >10% downside from D0 close before/through short re-acceleration, and +15% high/re-limit-up within D+10.
        idx10=np.arange(i+1,i+11)
        dd10=(np.nanmin(lo[idx10])/cl[i]-1)*100
        reup10=bool(np.any(hi[idx10]>=cl[i]*1.15) or np.any(cc[idx10]=="1"))
        type2=(not type1) and dd10>=-10.0 and reup10

        # 4: -10% close failure occurs before a +15% re-acceleration / another limit-up.
        type4=False
        for j in idx10:
            if hi[j]>=cl[i]*1.15 or cc[j]=="1":
                break
            if cl[j]<=cl[i]*0.90:
                type4=True
                break

        if type1:
            path_type=1
        elif type2:
            path_type=2
        elif type4:
            path_type=4
        else:
            longidx=np.arange(i+11,i+61)
            up=(np.nanmax(hi[longidx])/cl[i]-1)*100
            dn=(np.nanmin(lo[longidx])/cl[i]-1)*100
            path_type=31 if up>=15 else (32 if dn<=-15 else 30)

        tradable=(-5.0<=d1_gap<=3.0)
        tp_success=False
        mfe=mae=d5ret=np.nan
        if tradable:
            entry=op[i+1]
            hz=np.arange(i+1,i+6)
            mfe=(np.nanmax(hi[hz])/entry-1)*100
            mae=(np.nanmin(lo[hz])/entry-1)*100
            d5ret=(cl[i+5]/entry-1)*100
            # Conservative daily-bar ordering: if TP and close-stop coexist on a day, count failure.
            for j in hz:
                hit_tp=hi[j]>=entry*1.06
                hit_stop=cl[j]<=entry*0.97
                if hit_stop:
                    tp_success=False
                    break
                if hit_tp:
                    tp_success=True
                    break

        continuation=path_type in (1,2)
        positive=bool(tradable and continuation and tp_success)
        rec={"Code":code,"Name":nm[i],"Date":date,"year":int(date.year),"d1_gap":float(d1_gap),
             "path_type":path_type,"continuation":int(continuation),"failure":int(path_type==4),
             "tradable":int(tradable),"tp_success":int(tp_success),"positive":int(positive),
             "mfe":mfe,"mae":mae,"d5ret":d5ret}
        rec.update({k:float(v) for k,v in feature_vals.items()})
        events.append(rec)

ev=pd.DataFrame(events)
print("EVENTS",len(ev),ev.groupby("year").size().to_dict(),flush=True)
ev.to_csv("events_summary.csv",index=False)

features=base_features+["d1_gap"]

def wilson_lb(k,n,z=1.6448536269514722):
    if n<=0:return 0.0
    p=k/n
    den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def fit_model(X,y,kind):
    if kind=="logit":
        m=Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.35,max_iter=3000,class_weight="balanced",random_state=42))])
    else:
        m=HistGradientBoostingClassifier(max_depth=3,learning_rate=.04,max_iter=140,l2_regularization=2.0,min_samples_leaf=15,random_state=42)
    m.fit(X,y)
    return m

train=ev[(ev.year>=2015)&(ev.year<=2018)&(ev.tradable==1)].copy()
cal=ev[(ev.year==2019)&(ev.tradable==1)].copy()
test=ev[(ev.year>=2020)&(ev.year<=2026)&(ev.tradable==1)].copy()
print("SPLITS",len(train),len(cal),len(test),flush=True)
if min(len(train),len(cal),len(test))==0:
    raise RuntimeError("empty split")

X=train[features]
cont_l=fit_model(X,train.continuation,"logit")
cont_g=fit_model(X,train.continuation,"gb")
fail_l=fit_model(X,train.failure,"logit")
tp_g=fit_model(X,train.tp_success,"gb")

def raw_score(frame):
    X=frame[features]
    pc=.55*cont_l.predict_proba(X)[:,1]+.45*cont_g.predict_proba(X)[:,1]
    pf=fail_l.predict_proba(X)[:,1]
    pt=tp_g.predict_proba(X)[:,1]
    return np.clip(pc*(1-pf)*np.sqrt(np.clip(pt,1e-6,1)),1e-6,1-1e-6)

cal["raw"]=raw_score(cal)
platt=LogisticRegression(C=1.0,max_iter=2000,random_state=42).fit(cal[["raw"]],cal.positive)
cal["p"]=platt.predict_proba(cal[["raw"]])[:,1]

# Freeze selection threshold using 2019 only. No 2020+ outcomes participate.
opts=[]
for t in sorted(set(np.round(cal.p,4))):
    s=cal[cal.p>=t]
    n=len(s)
    if n<12:
        continue
    k=int(s.positive.sum())
    opts.append((wilson_lb(k,n),k/n,n,float(t)))
if not opts:
    raise RuntimeError("2019 calibration produced fewer than 12 selected cases")
cal_lb,cal_prec,cal_n,threshold=max(opts,key=lambda q:(q[0],q[1],q[2]))

test["raw"]=raw_score(test)
test["p"]=platt.predict_proba(test[["raw"]])[:,1]
sig=test[test.p>=threshold].copy()
N=len(sig); K=int(sig.positive.sum()); hit=100*K/N if N else 0.0

yearly=[]
for y in range(2020,2027):
    s=sig[sig.year==y]
    n=len(s); k=int(s.positive.sum())
    yearly.append({
        "year":y,"signals":n,"success":k,
        "hit_rate_pct":round(100*k/n,2) if n else None,
        "avg_mfe_pct":round(float(s.mfe.mean()),2) if n else None,
        "avg_mae_pct":round(float(s.mae.mean()),2) if n else None,
        "avg_d5_return_pct":round(float(s.d5ret.mean()),2) if n else None,
    })

path_counts={str(k):int(v) for k,v in sig.path_type.value_counts().to_dict().items()}
year_rule=all(x["signals"]==0 or x["hit_rate_pct"]>=70 for x in yearly)
result={
    "model":"KQ-LUCR v1.0 FIXED",
    "data_source":"FinanceData/marcap, derived from KRX daily all-stock data",
    "market":"KOSDAQ",
    "train_period":"2015-2018",
    "calibration_period":"2019",
    "test_period":"2020-2026",
    "upper_limit_event":"official KRX ChangeCode=1",
    "episode_dedupe":"first upper-limit event in a 5-trading-day cluster",
    "entry_rule":"D+1 open only when gap versus D0 close is -5% to +3%",
    "success_rule":"path type 1/2 AND +6% intraday high reached before any -3% close-stop within D+5; same-day close-stop overrides TP",
    "events_total":int(len(ev)),
    "splits":{"train_tradable":int(len(train)),"calibration_tradable":int(len(cal)),"test_tradable":int(len(test))},
    "frozen_threshold":float(threshold),
    "calibration":{"signals":int(cal_n),"success":int(round(cal_prec*cal_n)),"precision_pct":round(cal_prec*100,2),"one_sided_95_wilson_lb_pct":round(cal_lb*100,2)},
    "test":{
        "signals":int(N),"success":int(K),"hit_rate_pct":round(hit,2),
        "one_sided_95_wilson_lb_pct":round(wilson_lb(K,N)*100,2),
        "avg_mfe_pct":round(float(sig.mfe.mean()),2) if N else None,
        "avg_mae_pct":round(float(sig.mae.mean()),2) if N else None,
        "avg_d5_return_pct":round(float(sig.d5ret.mean()),2) if N else None,
        "path_type_counts":path_counts,
    },
    "yearly":yearly,
    "criteria":{"min_signals":100,"overall_hit_rate_min_pct":80.0,"yearly_hit_rate_min_pct":70.0},
    "pass_80":bool(N>=100 and hit>=80.0 and year_rule),
}
with open("result_ultra.json","w",encoding="utf-8") as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
sig[["Code","Name","Date","year","d1_gap","path_type","positive","mfe","mae","d5ret","p"]].to_csv("signals_ultra_2020_2026.csv",index=False)
print("FINAL_RESULT")
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
