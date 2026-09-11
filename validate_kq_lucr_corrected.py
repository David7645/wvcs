import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

YEARS=range(2015,2027)
COLS=["Date","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Market"]
frames=[]
for y in YEARS:
    p=f"data/marcap-{y}.parquet"
    x=pd.read_parquet(p,columns=COLS)
    x=x[x["Market"].astype(str).str.upper().eq("KOSDAQ")].copy()
    frames.append(x)
d=pd.concat(frames,ignore_index=True)
d["Date"]=pd.to_datetime(d["Date"])
d["Code"]=d["Code"].astype(str).str.zfill(6)
for c in ["Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap"]:
    d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna(subset=["Date","Code","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap"])
d=d[(d.Open>0)&(d.High>0)&(d.Low>0)&(d.Close>0)].copy()

# Correct upper-limit proxy from KRX daily fields:
# limit-up close is ~+30% vs official base price after tick-size truncation, and closes at daily high.
# ChangesRatio is KRX's official base-price-relative change, so it remains valid on corporate-action base-price days.
d["is_lu"]=(d["ChangesRatio"]>=29.5)&np.isclose(d["Close"],d["High"],rtol=0,atol=0)

# Exclude obvious SPACs from ordinary-equity continuation study.
d=d[~d["Name"].astype(str).str.contains("스팩",na=False)].copy()

breadth=d.groupby("Date",sort=False)["ChangesRatio"].apply(lambda s:float((s>0).mean())).to_dict()
medret=d.groupby("Date",sort=False)["ChangesRatio"].median().to_dict()
lucount=d[d["is_lu"]].groupby("Date").size().to_dict()

event_codes=set(d.loc[d["is_lu"],"Code"])
d=d[d["Code"].isin(event_codes)].sort_values(["Code","Date"]).reset_index(drop=True)
print("EVENT_CODES",len(event_codes),"ROWS",len(d),flush=True)

features0=["close_loc","ret5","ret20","ret60","accel","rv20","vol_ratio20","amt_ratio20",
"high20_dist","turnover_marcap","prior_lu60","marcap_log","amount_log","breadth","mkt_med_ret","limitup_count","gap0"]
events=[]

for code,gg in d.groupby("Code",sort=False):
    gg=gg.reset_index(drop=True)
    lu=gg["is_lu"].to_numpy(bool)
    inds=np.flatnonzero(lu)
    if len(inds)==0: continue
    op=gg.Open.to_numpy(float); hi=gg.High.to_numpy(float); lo=gg.Low.to_numpy(float); cl=gg.Close.to_numpy(float)
    vo=gg.Volume.to_numpy(float); am=gg.Amount.to_numpy(float); mc=gg.Marcap.to_numpy(float)
    dt=gg.Date.to_numpy(); nm=gg.Name.astype(str).to_numpy()
    last=-999
    for i in inds:
        if i-last<=4: continue
        last=i
        if i<60 or i+60>=len(gg): continue
        ret5=(cl[i]/cl[i-5]-1)*100; ret10=(cl[i]/cl[i-10]-1)*100
        ret20=(cl[i]/cl[i-20]-1)*100; ret60=(cl[i]/cl[i-60]-1)*100
        accel=ret5-ret10/2
        prev=cl[i-20:i+1]; rets=(prev[1:]/prev[:-1]-1)*100
        rv20=float(np.std(rets,ddof=1))
        vb=float(np.median(vo[i-20:i])); ab=float(np.median(am[i-20:i])); ph=float(np.max(hi[i-20:i]))
        date=pd.Timestamp(dt[i])
        fv={
          "close_loc":1.0 if hi[i]==lo[i] else (cl[i]-lo[i])/(hi[i]-lo[i]),
          "ret5":ret5,"ret20":ret20,"ret60":ret60,"accel":accel,"rv20":rv20,
          "vol_ratio20":vo[i]/vb if vb>0 else np.nan,"amt_ratio20":am[i]/ab if ab>0 else np.nan,
          "high20_dist":cl[i]/ph-1 if ph>0 else np.nan,"turnover_marcap":am[i]/mc[i] if mc[i]>0 else np.nan,
          "prior_lu60":float(np.sum(lu[i-60:i])),"marcap_log":math.log1p(mc[i]),"amount_log":math.log1p(am[i]),
          "breadth":float(breadth.get(date,np.nan)),"mkt_med_ret":float(medret.get(date,np.nan)),
          "limitup_count":float(lucount.get(date,0)),"gap0":(op[i]/cl[i-1]-1)*100
        }
        if any(not np.isfinite(v) for v in fv.values()): continue
        d1_gap=(op[i+1]/cl[i]-1)*100
        idx10=np.arange(i+1,i+11)
        type1=bool(lu[i+1])
        dd10=(np.min(lo[idx10])/cl[i]-1)*100
        reup10=bool(np.any(hi[idx10]>=cl[i]*1.15) or np.any(lu[idx10]))
        type2=(not type1) and dd10>=-10 and reup10
        type4=False
        for j in idx10:
            if hi[j]>=cl[i]*1.15 or lu[j]: break
            if cl[j]<=cl[i]*0.90:
                type4=True; break
        if type1: ptype=1
        elif type2: ptype=2
        elif type4: ptype=4
        else:
            idx60=np.arange(i+11,i+61)
            up=(np.max(hi[idx60])/cl[i]-1)*100; dn=(np.min(lo[idx60])/cl[i]-1)*100
            ptype=31 if up>=15 else (32 if dn<=-15 else 30)

        tradable=(-5<=d1_gap<=3)
        tp=False; mfe=mae=d5ret=np.nan
        if tradable:
            entry=op[i+1]; hz=np.arange(i+1,i+6)
            mfe=(np.max(hi[hz])/entry-1)*100; mae=(np.min(lo[hz])/entry-1)*100; d5ret=(cl[i+5]/entry-1)*100
            for j in hz:
                # stop is close-based; if same day TP and close-stop both occur, conservative fail
                if cl[j]<=entry*.97: tp=False; break
                if hi[j]>=entry*1.06: tp=True; break
        cont=ptype in (1,2)
        pos=bool(tradable and cont and tp)
        rec={"Code":code,"Name":nm[i],"Date":date,"year":int(date.year),"d1_gap":float(d1_gap),"path_type":ptype,
             "continuation":int(cont),"failure":int(ptype==4),"tradable":int(tradable),"tp_success":int(tp),"positive":int(pos),
             "mfe":mfe,"mae":mae,"d5ret":d5ret}
        rec.update({k:float(v) for k,v in fv.items()})
        events.append(rec)

ev=pd.DataFrame(events)
print("EVENTS_TOTAL",len(ev),flush=True)
print("EVENTS_BY_YEAR",ev.groupby("year").size().to_dict(),flush=True)

features=features0+["d1_gap"]
def lb(k,n,z=1.6448536269514722):
    if n<=0:return 0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den
def fit(X,y,kind):
    if kind=="logit":
        m=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.35,max_iter=3000,class_weight="balanced",random_state=42))])
    else:
        m=HistGradientBoostingClassifier(max_depth=3,learning_rate=.04,max_iter=140,l2_regularization=2,min_samples_leaf=12,random_state=42)
    m.fit(X,y); return m

train=ev[(ev.year>=2015)&(ev.year<=2018)&(ev.tradable==1)].copy()
cal=ev[(ev.year==2019)&(ev.tradable==1)].copy()
test=ev[(ev.year>=2020)&(ev.year<=2026)&(ev.tradable==1)].copy()
print("SPLITS",len(train),len(cal),len(test),flush=True)
X=train[features]
mc1=fit(X,train.continuation,"logit"); mc2=fit(X,train.continuation,"gb")
mf=fit(X,train.failure,"logit"); mt=fit(X,train.tp_success,"gb")
def raw(z):
    X=z[features]
    pc=.55*mc1.predict_proba(X)[:,1]+.45*mc2.predict_proba(X)[:,1]
    pf=mf.predict_proba(X)[:,1]; pt=mt.predict_proba(X)[:,1]
    return np.clip(pc*(1-pf)*np.sqrt(np.clip(pt,1e-6,1)),1e-6,1-1e-6)

cal["raw"]=raw(cal)
pl=LogisticRegression(C=1,max_iter=2000,random_state=42).fit(cal[["raw"]],cal.positive)
cal["p"]=pl.predict_proba(cal[["raw"]])[:,1]
opts=[]
for t in sorted(set(np.round(cal.p,4))):
    s=cal[cal.p>=t]; n=len(s)
    if n<12: continue
    k=int(s.positive.sum()); opts.append((lb(k,n),k/n,n,float(t)))
if not opts: raise RuntimeError("No calibration threshold with >=12 signals")
clb,cprec,cn,thr=max(opts,key=lambda x:(x[0],x[1],x[2]))
test["raw"]=raw(test); test["p"]=pl.predict_proba(test[["raw"]])[:,1]
sig=test[test.p>=thr].copy()
N=len(sig); K=int(sig.positive.sum()); hit=100*K/N if N else 0
yearly=[]
for y in range(2020,2027):
    s=sig[sig.year==y]; n=len(s); k=int(s.positive.sum())
    yearly.append({"year":y,"signals":n,"success":k,"hit_rate_pct":round(100*k/n,2) if n else None,
                   "avg_mfe_pct":round(float(s.mfe.mean()),2) if n else None,
                   "avg_mae_pct":round(float(s.mae.mean()),2) if n else None,
                   "avg_d5_return_pct":round(float(s.d5ret.mean()),2) if n else None})
result={
"model":"KQ-LUCR v1.0 CORRECTED FIXED",
"upper_limit_definition":"KRX ChangesRatio>=29.5 and Close==High; SPAC excluded",
"train":"2015-2018","calibration":"2019","test":"2020-2026",
"event_count":int(len(ev)),
"splits":{"train":len(train),"calibration":len(cal),"test":len(test)},
"threshold":thr,
"calibration":{"signals":cn,"success":int(round(cprec*cn)),"precision_pct":round(cprec*100,2),"wilson_lb_pct":round(clb*100,2)},
"test":{"signals":N,"success":K,"hit_rate_pct":round(hit,2),"wilson_lb_pct":round(lb(K,N)*100,2),
        "avg_mfe_pct":round(float(sig.mfe.mean()),2) if N else None,"avg_mae_pct":round(float(sig.mae.mean()),2) if N else None,
        "avg_d5_return_pct":round(float(sig.d5ret.mean()),2) if N else None,
        "path_type_counts":{str(k):int(v) for k,v in sig.path_type.value_counts().to_dict().items()}},
"yearly":yearly,
"pass_80":bool(N>=100 and hit>=80 and all(x["signals"]==0 or x["hit_rate_pct"]>=70 for x in yearly))
}
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
open("result_corrected.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
