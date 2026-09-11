import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED=42
YEARS=range(2015,2027)
COLS=["Date","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Market"]
frames=[]
for y in YEARS:
    x=pd.read_parquet(f"data/marcap-{y}.parquet",columns=COLS)
    x=x[x["Market"].astype(str).str.upper().eq("KOSDAQ")].copy()
    frames.append(x)
d=pd.concat(frames,ignore_index=True)
d["Date"]=pd.to_datetime(d["Date"]); d["Code"]=d["Code"].astype(str).str.zfill(6)
for c in ["Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap"]:
    d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna(subset=["Date","Code","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap"])
d=d[(d.Open>0)&(d.High>0)&(d.Low>0)&(d.Close>0)].copy()
d=d[~d["Name"].astype(str).str.contains("스팩",na=False)].copy()
d["is_lu"]=(d["ChangesRatio"]>=29.5)&(d["Close"]==d["High"])

# Market regime features available at D0 close.
day=d.groupby("Date",sort=True).agg(
    breadth=("ChangesRatio",lambda s:float((s>0).mean())),
    medret=("ChangesRatio","median"),
    lucount=("is_lu","sum"),
)
day["breadth5"]=day["breadth"].rolling(5,min_periods=3).mean()
day["breadth20"]=day["breadth"].rolling(20,min_periods=10).mean()
day["medret5"]=day["medret"].rolling(5,min_periods=3).mean()
day["medret20"]=day["medret"].rolling(20,min_periods=10).mean()
day["lucount5"]=day["lucount"].rolling(5,min_periods=3).mean()
day["lucount20"]=day["lucount"].rolling(20,min_periods=10).mean()
dm=day.to_dict("index")

event_codes=set(d.loc[d.is_lu,"Code"])
d=d[d.Code.isin(event_codes)].sort_values(["Code","Date"]).reset_index(drop=True)

events=[]
for code,gg in d.groupby("Code",sort=False):
    gg=gg.reset_index(drop=True)
    lu=gg.is_lu.to_numpy(bool); inds=np.flatnonzero(lu)
    if not len(inds): continue
    op=gg.Open.to_numpy(float); hi=gg.High.to_numpy(float); lo=gg.Low.to_numpy(float); cl=gg.Close.to_numpy(float)
    vo=gg.Volume.to_numpy(float); am=gg.Amount.to_numpy(float); mc=gg.Marcap.to_numpy(float)
    rr=gg.ChangesRatio.to_numpy(float); dt=gg.Date.to_numpy(); nm=gg.Name.astype(str).to_numpy()
    last=-999
    for i in inds:
        if i-last<=4: continue
        last=i
        if i<120 or i+60>=len(gg): continue
        date=pd.Timestamp(dt[i]); mk=dm.get(date,{})
        if not mk or any(pd.isna(mk.get(k,np.nan)) for k in ["breadth5","breadth20","medret5","medret20","lucount5","lucount20"]):
            continue
        # Pre-D0 returns and structure.
        def ret(n): return (cl[i]/cl[i-n]-1)*100
        ret1pre=(cl[i-1]/cl[i-2]-1)*100
        ret2pre=(cl[i-1]/cl[i-3]-1)*100
        ret5,ret10,ret20,ret60,ret120=[ret(n) for n in [5,10,20,60,120]]
        accel5=ret5-ret20/4.0
        accel10=ret10-ret60/6.0
        p20=cl[i-20:i+1]; p60=cl[i-60:i+1]
        rv10=float(np.std((cl[i-9:i+1]/cl[i-10:i]-1)*100,ddof=1))
        rv20=float(np.std((p20[1:]/p20[:-1]-1)*100,ddof=1))
        vb20=float(np.median(vo[i-20:i])); ab20=float(np.median(am[i-20:i]))
        vb5=float(np.median(vo[i-5:i])); ab5=float(np.median(am[i-5:i]))
        ph20=float(np.max(hi[i-20:i])); ph60=float(np.max(hi[i-60:i])); ph120=float(np.max(hi[i-120:i]))
        ma20=float(np.mean(cl[i-20:i])); ma60=float(np.mean(cl[i-60:i]))
        prevret20=(cl[i-19:i]/cl[i-20:i-1]-1)*100
        pre_maxret5=float(np.max((cl[i-4:i+1]/cl[i-5:i]-1)*100))
        pre_maxret20=float(np.max(prevret20))
        low_ret0=(lo[i]/cl[i-1]-1)*100
        open_ret0=(op[i]/cl[i-1]-1)*100
        range0=(hi[i]/lo[i]-1)*100 if lo[i]>0 else np.nan
        open_to_close=(cl[i]/op[i]-1)*100 if op[i]>0 else np.nan
        one_price=float(op[i]==hi[i]==lo[i]==cl[i])
        vals={
          "ret1pre":ret1pre,"ret2pre":ret2pre,"ret5":ret5,"ret10":ret10,"ret20":ret20,"ret60":ret60,"ret120":ret120,
          "accel5":accel5,"accel10":accel10,"rv10":rv10,"rv20":rv20,
          "vol_ratio20":vo[i]/vb20 if vb20>0 else np.nan,"amt_ratio20":am[i]/ab20 if ab20>0 else np.nan,
          "vol_ratio5":vo[i]/vb5 if vb5>0 else np.nan,"amt_ratio5":am[i]/ab5 if ab5>0 else np.nan,
          "turnover":am[i]/mc[i] if mc[i]>0 else np.nan,"marcap_log":math.log1p(mc[i]),"amount_log":math.log1p(am[i]),
          "high20_dist":cl[i]/ph20-1,"high60_dist":cl[i]/ph60-1,"high120_dist":cl[i]/ph120-1,
          "ma20_dist":cl[i]/ma20-1,"ma60_dist":cl[i]/ma60-1,
          "prior_lu20":float(np.sum(lu[i-20:i])),"prior_lu60":float(np.sum(lu[i-60:i])),"prior_lu120":float(np.sum(lu[i-120:i])),
          "pre_maxret5":pre_maxret5,"pre_maxret20":pre_maxret20,
          "gap0":open_ret0,"low_ret0":low_ret0,"range0":range0,"open_to_close0":open_to_close,"one_price":one_price,
          "breadth":mk["breadth"],"breadth5":mk["breadth5"],"breadth20":mk["breadth20"],
          "medret":mk["medret"],"medret5":mk["medret5"],"medret20":mk["medret20"],
          "lucount":mk["lucount"],"lucount5":mk["lucount5"],"lucount20":mk["lucount20"],
        }
        if any(not np.isfinite(v) for v in vals.values()): continue
        d1_gap=(op[i+1]/cl[i]-1)*100
        idx10=np.arange(i+1,i+11)
        type1=bool(lu[i+1])
        dd10=(np.min(lo[idx10])/cl[i]-1)*100
        reup10=bool(np.any(hi[idx10]>=cl[i]*1.15) or np.any(lu[idx10]))
        type2=(not type1) and dd10>=-10 and reup10
        type4=False
        for j in idx10:
            if hi[j]>=cl[i]*1.15 or lu[j]: break
            if cl[j]<=cl[i]*.90: type4=True; break
        if type1: ptype=1
        elif type2: ptype=2
        elif type4: ptype=4
        else:
            z=np.arange(i+11,i+61); up=(np.max(hi[z])/cl[i]-1)*100; dn=(np.min(lo[z])/cl[i]-1)*100
            ptype=31 if up>=15 else (32 if dn<=-15 else 30)
        tradable=(-5<=d1_gap<=3)
        tp=False; mfe=mae=d5ret=np.nan
        if tradable:
            entry=op[i+1]; hz=np.arange(i+1,i+6)
            mfe=(np.max(hi[hz])/entry-1)*100; mae=(np.min(lo[hz])/entry-1)*100; d5ret=(cl[i+5]/entry-1)*100
            for j in hz:
                if cl[j]<=entry*.97: tp=False; break
                if hi[j]>=entry*1.06: tp=True; break
        cont=ptype in (1,2); pos=bool(tradable and cont and tp)
        rec={"Code":code,"Name":nm[i],"Date":date,"year":date.year,"d1_gap":float(d1_gap),"path_type":ptype,
             "continuation":int(cont),"failure":int(ptype==4),"positive":int(pos),"tradable":int(tradable),
             "mfe":mfe,"mae":mae,"d5ret":d5ret}
        rec.update({k:float(v) for k,v in vals.items()}); events.append(rec)
ev=pd.DataFrame(events).sort_values("Date").reset_index(drop=True)

# Stock-specific Bayesian history features. Only prior labelled events for the same code are used.
for c in ["prior_ev_count","prior_cont_rate","prior_pos_rate","prior_fail_rate"]:
    ev[c]=0.0
state={}
for idx,r in ev.iterrows():
    st=state.get(r.Code,[0,0,0,0]) # n, cont, pos, fail
    n,co,po,fa=st
    ev.at[idx,"prior_ev_count"]=n
    # Jeffreys/Beta shrinkage toward 0.5, intentionally conservative at low n.
    ev.at[idx,"prior_cont_rate"]=(co+1.0)/(n+2.0)
    ev.at[idx,"prior_pos_rate"]=(po+1.0)/(n+2.0)
    ev.at[idx,"prior_fail_rate"]=(fa+1.0)/(n+2.0)
    state[r.Code]=[n+1,co+int(r.continuation),po+int(r.positive),fa+int(r.failure)]

features=[c for c in ev.columns if c not in ["Code","Name","Date","year","path_type","continuation","failure","positive","tradable","mfe","mae","d5ret"]]
# D+1 gap is intentionally included because actual decision is at next-day open.
features=[c for c in features if c!="d1_gap"]+["d1_gap"]

def make_models():
    return {
      "et":ExtraTreesClassifier(n_estimators=500,max_depth=7,min_samples_leaf=6,max_features=.7,class_weight="balanced",random_state=SEED,n_jobs=-1),
      "hgb":HistGradientBoostingClassifier(max_depth=4,learning_rate=.045,max_iter=180,l2_regularization=3.0,min_samples_leaf=12,random_state=SEED),
      "lr":Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.25,max_iter=3000,class_weight="balanced",random_state=SEED))])
    }

def fit_triplet(train):
    out={}
    for target in ["continuation","failure","positive"]:
        for name,m in make_models().items():
            mm=m
            mm.fit(train[features],train[target])
            out[f"{target}_{name}"]=mm
    return out

def pred_triplet(mods,z):
    X=z[features]
    p={}
    for key,m in mods.items():
        p[key]=m.predict_proba(X)[:,1]
    return pd.DataFrame(p,index=z.index)

# Walk-forward OOF base predictions for pre-2020 years.
oof_parts=[]
for vy in [2016,2017,2018,2019]:
    tr=ev[(ev.year<vy)&(ev.year>=2015)&(ev.tradable==1)]
    va=ev[(ev.year==vy)&(ev.tradable==1)]
    if len(tr)<50 or len(va)==0: continue
    mods=fit_triplet(tr)
    pp=pred_triplet(mods,va)
    tmp=va[["year","positive","continuation","failure","d1_gap","turnover","one_price","prior_cont_rate","prior_pos_rate","prior_fail_rate","breadth5","medret5","ret5","amt_ratio20"]].copy()
    tmp=tmp.join(pp)
    oof_parts.append(tmp)
oof=pd.concat(oof_parts).sort_index()

base_prob_cols=[c for c in oof.columns if c.startswith(("continuation_","failure_","positive_"))]
meta_extra=["d1_gap","turnover","one_price","prior_cont_rate","prior_pos_rate","prior_fail_rate","breadth5","medret5","ret5","amt_ratio20"]
# Add consensus/disagreement statistics as learned meta-features, not hand-summed decision rules.
def add_meta_stats(z):
    z=z.copy()
    for tgt in ["continuation","failure","positive"]:
        cols=[c for c in z.columns if c.startswith(tgt+"_")]
        z[tgt+"_mean"]=z[cols].mean(axis=1)
        z[tgt+"_std"]=z[cols].std(axis=1)
        z[tgt+"_min"]=z[cols].min(axis=1)
        z[tgt+"_max"]=z[cols].max(axis=1)
    return z
oof=add_meta_stats(oof)
meta_cols=base_prob_cols+meta_extra+[f"{t}_{s}" for t in ["continuation","failure","positive"] for s in ["mean","std","min","max"]]

# True stacked model: train meta learner only on OOF 2016-2018.
meta_train=oof[oof.year<=2018]
meta=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.15,max_iter=5000,class_weight="balanced",random_state=SEED))])
meta.fit(meta_train[meta_cols],meta_train.positive)

# 2019 is dedicated threshold-selection set; no 2020+ labels used.
cal=oof[oof.year==2019].copy()
cal["score"]=meta.predict_proba(cal[meta_cols])[:,1]

def wilson(k,n,z=1.6448536269514722):
    if n<=0:return 0.0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

opts=[]
# Coarse quantile family to reduce threshold overfit. Require >=12 2019 signals.
for q in np.arange(.70,.971,.025):
    t=float(cal.score.quantile(q))
    s=cal[cal.score>=t]; n=len(s); k=int(s.positive.sum())
    if n>=12: opts.append((wilson(k,n),k/n,n,q,t))
if not opts: raise RuntimeError("no calibration candidate")
cal_lb,cal_prec,cal_n,qstar,tstar=max(opts,key=lambda z:(z[0],z[1],z[2]))

# Freeze at 2020 boundary: base models trained only through 2018, matching the 2019 calibration generation.
final_train=ev[(ev.year>=2015)&(ev.year<=2018)&(ev.tradable==1)]
mods=fit_triplet(final_train)
test=ev[(ev.year>=2020)&(ev.year<=2026)&(ev.tradable==1)].copy()
pp=pred_triplet(mods,test)
mz=test[["d1_gap","turnover","one_price","prior_cont_rate","prior_pos_rate","prior_fail_rate","breadth5","medret5","ret5","amt_ratio20"]].copy().join(pp)
mz=add_meta_stats(mz)
test["score"]=meta.predict_proba(mz[meta_cols])[:,1]
sig=test[test.score>=tstar].copy()
N=len(sig);K=int(sig.positive.sum());hit=100*K/N if N else 0
contK=int(sig.continuation.sum()); cont_prec=100*contK/N if N else 0
yearly=[]
for y in range(2020,2027):
    s=sig[sig.year==y]; n=len(s); k=int(s.positive.sum())
    yearly.append({"year":y,"signals":n,"success":k,"hit_rate_pct":round(100*k/n,2) if n else None,
                   "continuation_pct":round(100*s.continuation.mean(),2) if n else None,
                   "avg_mfe_pct":round(float(s.mfe.mean()),2) if n else None,
                   "avg_mae_pct":round(float(s.mae.mean()),2) if n else None,
                   "avg_d5_return_pct":round(float(s.d5ret.mean()),2) if n else None})
result={
 "model":"SANGTTA-STACK v2.0 FIXED",
 "method":"walk-forward OOF stacked selective classifier; base experts for continuation/failure/trade-success; logistic meta learner",
 "data":"FinanceData/marcap KOSDAQ; upper-limit=ChangesRatio>=29.5 & Close==High; SPAC excluded",
 "development":{"base_oof_years":"2016-2019","meta_train_oof":"2016-2018","threshold_calibration":"2019","quantile_family":"0.70..0.95 step 0.025","min_2019_signals":12},
 "test":"2020-2026 untouched",
 "events_total":int(len(ev)),
 "tradable_counts":{"train_2015_2018":int(len(final_train)),"cal_2019":int(len(cal)),"test_2020_2026":int(len(test))},
 "threshold":{"quantile":round(float(qstar),3),"score":round(float(tstar),6)},
 "calibration":{"signals":int(cal_n),"success":int(round(cal_prec*cal_n)),"hit_rate_pct":round(cal_prec*100,2),"wilson_lb_pct":round(cal_lb*100,2)},
 "test_result":{"signals":N,"success":K,"hit_rate_pct":round(hit,2),"wilson_lb_pct":round(wilson(K,N)*100,2),
                "continuation_precision_pct":round(cont_prec,2),
                "avg_mfe_pct":round(float(sig.mfe.mean()),2) if N else None,
                "avg_mae_pct":round(float(sig.mae.mean()),2) if N else None,
                "avg_d5_return_pct":round(float(sig.d5ret.mean()),2) if N else None,
                "path_type_counts":{str(k):int(v) for k,v in sig.path_type.value_counts().to_dict().items()}},
 "yearly":yearly,
 "pass_80":bool(N>=100 and hit>=80 and all(x["signals"]==0 or x["hit_rate_pct"]>=70 for x in yearly))
}
print("RESEARCH_EVENTS",len(ev),flush=True)
print("CALIBRATION_CANDIDATES",[(round(a*100,1),round(b*100,1),n,round(q,3)) for a,b,n,q,t in sorted(opts,reverse=True)[:5]],flush=True)
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
open("stack_v2_result.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
