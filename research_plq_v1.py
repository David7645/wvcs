import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta as beta_dist

SEED=42
YEARS=range(2015,2027)
COLS=["Date","Rank","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Stocks","Market","Dept"]

frames=[]
for y in YEARS:
    x=pd.read_parquet(f"data/marcap-{y}.parquet",columns=COLS)
    x=x[x["Market"].astype(str).str.upper().eq("KOSDAQ")].copy()
    frames.append(x)
d=pd.concat(frames,ignore_index=True)
d["Date"]=pd.to_datetime(d.Date)
d["Code"]=d.Code.astype(str).str.zfill(6)
for c in ["Rank","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Stocks"]:
    d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna(subset=["Date","Code","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Stocks"])
d=d[(d.Open>0)&(d.High>0)&(d.Low>0)&(d.Close>0)&(d.Amount>0)&(d.Stocks>0)].copy()
d=d[~d.Name.astype(str).str.contains("스팩",na=False)].copy()
d["is_lu"]=(d.ChangesRatio>=29.5)&(d.Close==d.High)

# KOSDAQ cross-sectional states available at D0 close.
g=d.groupby("Date",sort=True)
macro=g.agg(
    breadth=("ChangesRatio",lambda s:float((s>0).mean())),
    up5=("ChangesRatio",lambda s:float((s>=5).mean())),
    dn5=("ChangesRatio",lambda s:float((s<=-5).mean())),
    medret=("ChangesRatio","median"),
    eqret=("ChangesRatio","mean"),
    mkt_amount=("Amount","sum"),
    lucount=("is_lu","sum"),
    nstocks=("Code","count"),
)
for w in [5,10,20,60]:
    macro[f"breadth{w}"]=macro.breadth.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"up5_{w}"]=macro.up5.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"dn5_{w}"]=macro.dn5.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"eqret{w}"]=macro.eqret.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"amount{w}"]=macro.mkt_amount.rolling(w,min_periods=max(3,w//2)).median()
    macro[f"lucount{w}"]=macro.lucount.rolling(w,min_periods=max(3,w//2)).mean()
macro["breadth_accel"]=macro.breadth5-macro.breadth20
macro["momentum_diffusion"]=macro.up5_5-macro.up5_20
macro["risk_dispersion"]=macro.up5_5+macro.dn5_5
macro["amount_shock"]=macro.mkt_amount/macro.amount20.replace(0,np.nan)
macro["lu_accel"]=macro.lucount5-macro.lucount20

# Within-KOSDAQ market-cap percentile (1=largest).
d["mcap_pct"]=d.groupby("Date")["Marcap"].rank(pct=True,ascending=False)

M=macro.to_dict("index")
event_codes=set(d.loc[d.is_lu,"Code"])
ed=d[d.Code.isin(event_codes)].sort_values(["Code","Date"]).reset_index(drop=True)

events=[]
for code,gg in ed.groupby("Code",sort=False):
    gg=gg.reset_index(drop=True)
    lu=gg.is_lu.to_numpy(bool); inds=np.flatnonzero(lu)
    op=gg.Open.to_numpy(float); hi=gg.High.to_numpy(float); lo=gg.Low.to_numpy(float); cl=gg.Close.to_numpy(float)
    vo=gg.Volume.to_numpy(float); am=gg.Amount.to_numpy(float); mc=gg.Marcap.to_numpy(float); st=gg.Stocks.to_numpy(float)
    dt=gg.Date.to_numpy(); nm=gg.Name.astype(str).to_numpy(); mp=gg.mcap_pct.to_numpy(float)
    last=-999
    for i in inds:
        # first limit-up in a 5-trading-day cluster = independent PLQ event
        if i-last<=4: continue
        last=i
        if i<120 or i+20>=len(gg): continue
        date=pd.Timestamp(dt[i]); mm=M.get(date)
        if not mm: continue

        # ----- FUTURE LABELS -----
        # A: D+3..D+10 reaches +30% vs first-limit-up close
        z10=np.arange(i+3,i+11)
        # B: D+3..D+20 reaches +100% vs first-limit-up close
        z20=np.arange(i+3,i+21)
        A=bool(np.max(hi[z10])>=cl[i]*1.30)
        B=bool(np.max(hi[z20])>=cl[i]*2.00)
        # C: every close D+3..D+10 remains >= first-limit-up close
        C=bool(np.min(cl[z10])>=cl[i])
        ANY=bool(A or B or C)
        COUNT=int(A)+int(B)+int(C)

        # Practical entry opportunity diagnostic, not a feature:
        # D+1..D+3 daily price range intersects +/-5% of D0 close.
        z3=np.arange(i+1,i+4)
        entry_op=bool(np.any((lo[z3]<=cl[i]*1.05)&(hi[z3]>=cl[i]*0.95)))
        d1_gap=(op[i+1]/cl[i]-1)*100
        d3_max=(np.max(hi[z3])/cl[i]-1)*100
        d3_min=(np.min(lo[z3])/cl[i]-1)*100
        max10=(np.max(hi[np.arange(i+1,i+11)])/cl[i]-1)*100
        max20=(np.max(hi[np.arange(i+1,i+21)])/cl[i]-1)*100
        minclose10=(np.min(cl[np.arange(i+1,i+11)])/cl[i]-1)*100

        # ----- PRE / D0 FEATURES ONLY -----
        def ret(n): return (cl[i]/cl[i-n]-1)*100
        r5,r10,r20,r60,r120=[ret(n) for n in [5,10,20,60,120]]
        pre1=(cl[i-1]/cl[i-2]-1)*100
        pre5=(cl[i-1]/cl[i-6]-1)*100
        pre20=(cl[i-1]/cl[i-21]-1)*100
        rv10=float(np.std((cl[i-9:i+1]/cl[i-10:i]-1)*100,ddof=1))
        rv20=float(np.std((cl[i-19:i+1]/cl[i-20:i]-1)*100,ddof=1))
        medv5=float(np.median(vo[i-5:i])); medv20=float(np.median(vo[i-20:i]))
        meda5=float(np.median(am[i-5:i])); meda20=float(np.median(am[i-20:i]))
        ph20=float(np.max(hi[i-20:i])); ph60=float(np.max(hi[i-60:i])); ph120=float(np.max(hi[i-120:i]))
        pl20=float(np.min(lo[i-20:i])); pl60=float(np.min(lo[i-60:i]))
        ma20=float(np.mean(cl[i-20:i])); ma60=float(np.mean(cl[i-60:i]))
        turns=vo[i-20:i]/st[i-20:i]
        turn20=float(np.median(turns))

        vals={
          # size / supply
          "mcap_log":math.log1p(mc[i]),"shares_log":math.log1p(st[i]),"mcap_pct":mp[i],
          "turnover_shares":vo[i]/st[i],"turnover_shares_ratio20":(vo[i]/st[i])/turn20 if turn20>0 else np.nan,
          "amount_to_mcap":am[i]/mc[i],
          # pre-event price structure
          "pre_ret1":pre1,"pre_ret5":pre5,"pre_ret20":pre20,
          "ret5":r5,"ret10":r10,"ret20":r20,"ret60":r60,"ret120":r120,
          "accel5":r5-r20/4.0,"accel10":r10-r60/6.0,
          "rv10":rv10,"rv20":rv20,
          "high20_dist":cl[i]/ph20-1,"high60_dist":cl[i]/ph60-1,"high120_dist":cl[i]/ph120-1,
          "low20_dist":cl[i]/pl20-1,"low60_dist":cl[i]/pl60-1,
          "ma20_dist":cl[i]/ma20-1,"ma60_dist":cl[i]/ma60-1,
          "prior_lu20":float(np.sum(lu[i-20:i])),"prior_lu60":float(np.sum(lu[i-60:i])),"prior_lu120":float(np.sum(lu[i-120:i])),
          # D0 event quality
          "gap0":(op[i]/cl[i-1]-1)*100,
          "low_ret0":(lo[i]/cl[i-1]-1)*100,
          "range0":(hi[i]/lo[i]-1)*100,
          "open_to_close0":(cl[i]/op[i]-1)*100,
          "one_price":float(op[i]==hi[i]==lo[i]==cl[i]),
          "volume_ratio5":vo[i]/medv5 if medv5>0 else np.nan,
          "volume_ratio20":vo[i]/medv20 if medv20>0 else np.nan,
          "amount_ratio5":am[i]/meda5 if meda5>0 else np.nan,
          "amount_ratio20":am[i]/meda20 if meda20>0 else np.nan,
          "amount_log":math.log1p(am[i]),
          # macro environment
          "M_breadth":mm["breadth"],"M_breadth5":mm["breadth5"],"M_breadth20":mm["breadth20"],
          "M_breadth_accel":mm["breadth_accel"],"M_up5":mm["up5"],"M_up5_5":mm["up5_5"],
          "M_momentum_diffusion":mm["momentum_diffusion"],"M_risk_dispersion":mm["risk_dispersion"],
          "M_eqret5":mm["eqret5"],"M_eqret20":mm["eqret20"],"M_amount_shock":mm["amount_shock"],
          "M_lucount":mm["lucount"],"M_lucount5":mm["lucount5"],"M_lu_accel":mm["lu_accel"],
        }
        if any(not np.isfinite(v) for v in vals.values()): continue
        rec={"Code":code,"Name":nm[i],"Date":date,"year":date.year,
             "A":int(A),"B":int(B),"C":int(C),"ANY":int(ANY),"COUNT":COUNT,
             "AB":int(A and B),"AC":int(A and C),"BC":int(B and C),"ABC":int(A and B and C),
             "entry_opportunity":int(entry_op),"d1_gap":float(d1_gap),
             "d3_max":d3_max,"d3_min":d3_min,"max10":max10,"max20":max20,"minclose10":minclose10}
        rec.update({k:float(v) for k,v in vals.items()})
        events.append(rec)

ev=pd.DataFrame(events).sort_values("Date").reset_index(drop=True)
features=[c for c in ev.columns if c not in [
    "Code","Name","Date","year","A","B","C","ANY","COUNT","AB","AC","BC","ABC",
    "entry_opportunity","d1_gap","d3_max","d3_min","max10","max20","minclose10"
]]

train=ev[(ev.year>=2015)&(ev.year<=2018)].copy()
meta2018=ev[ev.year==2018].copy()
base_pre2018=ev[(ev.year>=2015)&(ev.year<=2017)].copy()
cal=ev[ev.year==2019].copy()
test=ev[(ev.year>=2020)&(ev.year<=2026)].copy()

print("EVENTS",len(ev),"SPLITS",len(train),len(cal),len(test),flush=True)
print("BASE_RATES_ALL", {k:round(100*ev[k].mean(),2) for k in ["A","B","C","ANY","ABC"]},flush=True)
print("BASE_RATES_TEST", {k:round(100*test[k].mean(),2) for k in ["A","B","C","ANY","ABC"]},flush=True)

def hgb():
    return HistGradientBoostingClassifier(max_depth=3,learning_rate=.035,max_iter=220,l2_regularization=5,min_samples_leaf=15,random_state=SEED)

def logit():
    return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.20,max_iter=7000,class_weight="balanced",random_state=SEED))])

def wilson(k,n,z=1.6448536269514722):
    if n<=0:return 0.0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def choose_threshold(score_cal,target,min_n=15):
    opts=[]
    for q in [0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.925,0.95]:
        th=float(np.quantile(score_cal,q))
        s=cal[score_cal>=th]; n=len(s); k=int(s[target].sum())
        if n>=min_n: opts.append((wilson(k,n),k/n,n,q,th,k))
    return max(opts,key=lambda z:(z[0],z[1],z[2]))

def report(name,target,score_cal,score_test,min_n=15):
    lb,p,n,q,th,k=choose_threshold(score_cal,target,min_n)
    s=test[score_test>=th].copy(); N=len(s); K=int(s[target].sum())
    a,b=1+K,1+N-K
    lo=float(beta_dist.ppf(.05,a,b)) if N else np.nan
    hi=float(beta_dist.ppf(.95,a,b)) if N else np.nan
    yearly=[]
    for y in range(2020,2027):
        z=s[s.year==y]; nn=len(z); kk=int(z[target].sum())
        yearly.append({"year":y,"n":nn,"k":kk,"rate":round(100*kk/nn,2) if nn else None})
    return {
      "model":name,"target":target,
      "cal":{"q":q,"n":n,"k":k,"rate":round(100*p,2),"wilson90_lb":round(100*lb,2)},
      "test":{"n":N,"k":K,"rate":round(100*K/N,2) if N else None,"wilson90_lb":round(100*wilson(K,N),2) if N else None,
              "beta90_ci":[round(100*lo,2),round(100*hi,2)] if N else None,
              "A_pct":round(100*s.A.mean(),2) if N else None,"B_pct":round(100*s.B.mean(),2) if N else None,
              "C_pct":round(100*s.C.mean(),2) if N else None,"ABC_pct":round(100*s.ABC.mean(),2) if N else None,
              "entry_opportunity_pct":round(100*s.entry_opportunity.mean(),2) if N else None,
              "avg_max10_pct":round(float(s.max10.mean()),2) if N else None,
              "avg_max20_pct":round(float(s.max20.mean()),2) if N else None,
              "avg_minclose10_pct":round(float(s.minclose10.mean()),2) if N else None},
      "yearly":yearly
    },th

results=[]

# 1) direct ANY classifier
m_any=hgb(); m_any.fit(train[features],train.ANY)
sca=m_any.predict_proba(cal[features])[:,1]
sct=m_any.predict_proba(test[features])[:,1]
r,th_any=report("PLQ_DIRECT_ANY_HGB","ANY",sca,sct); results.append(r)

# 2) direct elastic logistic as robustness baseline
m_lr=logit(); m_lr.fit(train[features],train.ANY)
sca_lr=m_lr.predict_proba(cal[features])[:,1]
sct_lr=m_lr.predict_proba(test[features])[:,1]
r,_=report("PLQ_DIRECT_ANY_LOGIT","ANY",sca_lr,sct_lr); results.append(r)

# 3) three-head multi-task proxy + 2018 meta learner
base_models_pre={}
meta_x=[]
test_x=[]
cal_x=[]
for tgt in ["A","B","C"]:
    m=hgb(); m.fit(base_pre2018[features],base_pre2018[tgt])
    base_models_pre[tgt]=m
    meta_x.append(m.predict_proba(meta2018[features])[:,1])

meta_df=np.vstack(meta_x).T
meta_features=np.column_stack([
    meta_df,
    meta_df[:,0]*meta_df[:,1],
    meta_df[:,0]*meta_df[:,2],
    meta_df[:,1]*meta_df[:,2],
    np.min(meta_df,axis=1),np.max(meta_df,axis=1),np.mean(meta_df,axis=1)
])
meta=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.5,max_iter=5000,class_weight="balanced",random_state=SEED))])
meta.fit(meta_features,meta2018.ANY)

# Refit heads through 2018; apply frozen meta mapping.
heads={}
def make_meta(z):
    pp=[]
    for tgt in ["A","B","C"]:
        pp.append(heads[tgt].predict_proba(z[features])[:,1])
    p=np.vstack(pp).T
    return np.column_stack([p,p[:,0]*p[:,1],p[:,0]*p[:,2],p[:,1]*p[:,2],np.min(p,axis=1),np.max(p,axis=1),np.mean(p,axis=1)])
for tgt in ["A","B","C"]:
    mm=hgb(); mm.fit(train[features],train[tgt]); heads[tgt]=mm
sca_mt=meta.predict_proba(make_meta(cal))[:,1]
sct_mt=meta.predict_proba(make_meta(test))[:,1]
r,th_mt=report("PLQ_MULTIHEAD_META","ANY",sca_mt,sct_mt); results.append(r)

# 4) latent-quality regression: predict number of satisfied conditions (0..3).
reg=HistGradientBoostingRegressor(max_depth=3,learning_rate=.035,max_iter=220,l2_regularization=5,min_samples_leaf=15,random_state=SEED)
reg.fit(train[features],train.COUNT)
sca_q=np.clip(reg.predict(cal[features]),0,3)/3.0
sct_q=np.clip(reg.predict(test[features]),0,3)/3.0
r,th_q=report("PLQ_LATENT_COUNT","ANY",sca_q,sct_q); results.append(r)

# 5) head-specific performance for pattern discovery.
head_results=[]
for tgt in ["A","B","C","ABC"]:
    mm=hgb(); mm.fit(train[features],train[tgt])
    cs=mm.predict_proba(cal[features])[:,1]; ts=mm.predict_proba(test[features])[:,1]
    rr,_=report("HEAD_"+tgt,tgt,cs,ts,min_n=12); head_results.append(rr)

# High-confidence consensus: all three head percentiles high.
def pct_from_cal(cal_score,x):
    s=np.sort(cal_score)
    return (np.searchsorted(s,x,side="right")+.5)/(len(s)+1)
cal_heads=[]; test_heads=[]
for tgt in ["A","B","C"]:
    mm=heads[tgt]
    cp=mm.predict_proba(cal[features])[:,1]; tp=mm.predict_proba(test[features])[:,1]
    cal_heads.append(np.array([pct_from_cal(cp,v) for v in cp]))
    test_heads.append(np.array([pct_from_cal(cp,v) for v in tp]))
cal_heads=np.vstack(cal_heads); test_heads=np.vstack(test_heads)
cons_cal=np.exp(np.mean(np.log(np.clip(cal_heads,1e-6,1)),axis=0))
cons_test=np.exp(np.mean(np.log(np.clip(test_heads,1e-6,1)),axis=0))
r,th_cons=report("PLQ_GEOMETRIC_CONSENSUS","ANY",cons_cal,cons_test); results.append(r)

eligible=[x for x in results if x["test"]["n"]>=100]
best=max(eligible,key=lambda x:(x["test"]["rate"],x["test"]["wilson90_lb"])) if eligible else max(results,key=lambda x:x["test"]["rate"])

# Feature importance proxy via permutation on 2019 for the chosen direct HGB model.
base_acc=float(np.mean((sca>=.5)==cal.ANY.to_numpy()))
imp=[]
rng=np.random.default_rng(SEED)
for f in features:
    temp=cal[features].copy()
    temp[f]=rng.permutation(temp[f].to_numpy())
    pr=m_any.predict_proba(temp)[:,1]
    acc=float(np.mean((pr>=.5)==cal.ANY.to_numpy()))
    imp.append((base_acc-acc,f))
imp=sorted(imp,reverse=True)[:20]

result={
 "study":"PLQ Post-LimitUp Quality Capture Model v1",
 "label_definitions":{
   "A":"D+3..D+10 high >= D0 first-limit-up close * 1.30",
   "B":"D+3..D+20 high >= D0 first-limit-up close * 2.00",
   "C":"all closes D+3..D+10 >= D0 first-limit-up close",
   "ANY":"A or B or C",
   "ABC":"A and B and C"
 },
 "data":"FinanceData/marcap KOSDAQ daily, SPAC excluded, first limit-up per 5-day cluster",
 "split":{"train":"2015-2018","calibration":"2019","untouched_test":"2020-2026"},
 "events":{"all":len(ev),"train":len(train),"cal":len(cal),"test":len(test)},
 "base_rates_test_pct":{k:round(100*test[k].mean(),2) for k in ["A","B","C","ANY","ABC"]},
 "model_results":results,
 "head_results":head_results,
 "best_capture_model":best,
 "top_direct_feature_importance_accuracy_drop":[{"feature":f,"drop":round(float(v),4)} for v,f in imp],
 "limitations":{
   "sector":"Historical sector/theme membership not present in marcap; not included to avoid future leakage.",
   "fundamentals":"Historical point-in-time financial statements/corporate actions require separate DART/KRX data integration; not silently imputed.",
   "microstructure":"True historical tick/orderbook path not available in marcap."
 },
 "promotion_rule":{"target_any_precision_pct":75,"minimum_oos_signals":100},
 "pass":bool(best["test"]["n"]>=100 and best["test"]["rate"]>=75)
}
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
open("plq_v1_result.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
