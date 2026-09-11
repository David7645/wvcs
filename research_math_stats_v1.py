import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer, QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from scipy.stats import beta as beta_dist, binomtest

SEED=42
YEARS=range(2015,2027)
COLS=["Date","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Market"]
frames=[]
for y in YEARS:
    x=pd.read_parquet(f"data/marcap-{y}.parquet",columns=COLS)
    x=x[x["Market"].astype(str).str.upper().eq("KOSDAQ")].copy()
    frames.append(x)
d=pd.concat(frames,ignore_index=True)
d["Date"]=pd.to_datetime(d.Date); d["Code"]=d.Code.astype(str).str.zfill(6)
for c in ["Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap"]:
    d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna(subset=["Date","Code","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap"])
d=d[(d.Open>0)&(d.High>0)&(d.Low>0)&(d.Close>0)]
d=d[~d.Name.astype(str).str.contains("스팩",na=False)].copy()
d["is_lu"]=(d.ChangesRatio>=29.5)&(d.Close==d.High)

# ---------- MACRO state from all KOSDAQ stocks ----------
gday=d.groupby("Date",sort=True)
macro=gday.agg(
    breadth=("ChangesRatio",lambda s:float((s>0).mean())),
    medret=("ChangesRatio","median"),
    eqret=("ChangesRatio","mean"),
    mkt_amount=("Amount","sum"),
    lucount=("is_lu","sum"),
    up5=("ChangesRatio",lambda s:float((s>=5).mean())),
    dn5=("ChangesRatio",lambda s:float((s<=-5).mean())),
)
for w in [3,5,10,20,60]:
    macro[f"breadth{w}"]=macro.breadth.rolling(w,min_periods=max(2,w//2)).mean()
    macro[f"eqret{w}"]=macro.eqret.rolling(w,min_periods=max(2,w//2)).mean()
    macro[f"mkt_amt_ratio{w}"]=macro.mkt_amount/(macro.mkt_amount.shift(1).rolling(w,min_periods=max(2,w//2)).median())
    macro[f"lucount{w}"]=macro.lucount.rolling(w,min_periods=max(2,w//2)).mean()
    macro[f"up5_{w}"]=macro.up5.rolling(w,min_periods=max(2,w//2)).mean()
    macro[f"dn5_{w}"]=macro.dn5.rolling(w,min_periods=max(2,w//2)).mean()
macro["breadth_accel"]=macro.breadth5-macro.breadth20
macro["eqret_accel"]=macro.eqret5-macro.eqret20
macro["lu_accel"]=macro.lucount5-macro.lucount20
macro["risk_dispersion"]=macro.up5_5+macro.dn5_5
M=macro.to_dict("index")

# ---------- Event/meso/micro-proxy features ----------
event_codes=set(d.loc[d.is_lu,"Code"])
d=d[d.Code.isin(event_codes)].sort_values(["Code","Date"]).reset_index(drop=True)
events=[]
for code,gg in d.groupby("Code",sort=False):
    gg=gg.reset_index(drop=True)
    lu=gg.is_lu.to_numpy(bool); inds=np.flatnonzero(lu)
    op=gg.Open.to_numpy(float); hi=gg.High.to_numpy(float); lo=gg.Low.to_numpy(float); cl=gg.Close.to_numpy(float)
    vo=gg.Volume.to_numpy(float); am=gg.Amount.to_numpy(float); mc=gg.Marcap.to_numpy(float)
    dt=gg.Date.to_numpy(); nm=gg.Name.astype(str).to_numpy()
    last=-999
    for i in inds:
        if i-last<=4: continue
        last=i
        if i<120 or i+60>=len(gg): continue
        date=pd.Timestamp(dt[i]); mm=M.get(date)
        if not mm: continue
        macro_feats={k:float(v) for k,v in mm.items() if k!="mkt_amount"}
        if any(not np.isfinite(v) for v in macro_feats.values()): continue

        def ret(n): return (cl[i]/cl[i-n]-1)*100
        r5,r10,r20,r60,r120=[ret(n) for n in [5,10,20,60,120]]
        prevret20=(cl[i-19:i]/cl[i-20:i-1]-1)*100
        rv10=float(np.std((cl[i-9:i+1]/cl[i-10:i]-1)*100,ddof=1))
        rv20=float(np.std((cl[i-19:i+1]/cl[i-20:i]-1)*100,ddof=1))
        medv5=np.median(vo[i-5:i]); medv20=np.median(vo[i-20:i]); meda5=np.median(am[i-5:i]); meda20=np.median(am[i-20:i])
        ph20=np.max(hi[i-20:i]); ph60=np.max(hi[i-60:i]); ph120=np.max(hi[i-120:i])
        ma20=np.mean(cl[i-20:i]); ma60=np.mean(cl[i-60:i])
        meso={
          "ret5":r5,"ret10":r10,"ret20":r20,"ret60":r60,"ret120":r120,
          "accel5":r5-r20/4,"accel10":r10-r60/6,"rv10":rv10,"rv20":rv20,
          "high20_dist":cl[i]/ph20-1,"high60_dist":cl[i]/ph60-1,"high120_dist":cl[i]/ph120-1,
          "ma20_dist":cl[i]/ma20-1,"ma60_dist":cl[i]/ma60-1,
          "prior_lu20":float(np.sum(lu[i-20:i])),"prior_lu60":float(np.sum(lu[i-60:i])),"prior_lu120":float(np.sum(lu[i-120:i])),
          "marcap_log":math.log1p(mc[i]),"pre_maxret20":float(np.max(prevret20)),
        }
        micro={
          "gap0":(op[i]/cl[i-1]-1)*100,
          "low_ret0":(lo[i]/cl[i-1]-1)*100,
          "range0":(hi[i]/lo[i]-1)*100,
          "open_to_close0":(cl[i]/op[i]-1)*100,
          "one_price":float(op[i]==hi[i]==lo[i]==cl[i]),
          "vol_ratio5":vo[i]/medv5 if medv5>0 else np.nan,
          "vol_ratio20":vo[i]/medv20 if medv20>0 else np.nan,
          "amt_ratio5":am[i]/meda5 if meda5>0 else np.nan,
          "amt_ratio20":am[i]/meda20 if meda20>0 else np.nan,
          "turnover":am[i]/mc[i] if mc[i]>0 else np.nan,
          "amount_log":math.log1p(am[i]),
          "close_loc":1.0 if hi[i]==lo[i] else (cl[i]-lo[i])/(hi[i]-lo[i]),
        }
        if any(not np.isfinite(v) for v in list(meso.values())+list(micro.values())): continue

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
        cont=ptype in (1,2); fail=(ptype==4); pos=bool(tradable and cont and tp)
        rec={"Code":code,"Name":nm[i],"Date":date,"year":date.year,"d1_gap":float(d1_gap),"path_type":ptype,
             "continuation":int(cont),"failure":int(fail),"positive":int(pos),"tradable":int(tradable),"mfe":mfe,"mae":mae,"d5ret":d5ret}
        rec.update({"M_"+k:v for k,v in macro_feats.items()})
        rec.update({"S_"+k:v for k,v in meso.items()})
        rec.update({"X_"+k:v for k,v in micro.items()})
        events.append(rec)
ev=pd.DataFrame(events).sort_values("Date").reset_index(drop=True)

macro_cols=[c for c in ev.columns if c.startswith("M_")]
meso_cols=[c for c in ev.columns if c.startswith("S_")]
micro_cols=[c for c in ev.columns if c.startswith("X_")]+["d1_gap"]
all_cols=macro_cols+meso_cols+micro_cols

# Explicit macro x micro interactions to test user's core hypothesis.
interaction_pairs=[
 ("M_breadth5","X_turnover"),("M_breadth20","X_turnover"),("M_breadth_accel","X_turnover"),
 ("M_lucount5","X_amt_ratio20"),("M_lu_accel","X_amt_ratio20"),("M_eqret5","X_one_price"),
 ("M_up5_5","X_turnover"),("M_risk_dispersion","X_range0"),("M_breadth5","d1_gap"),
]
for a,b in interaction_pairs:
    if a in ev and b in ev:
        ev["I_"+a+"__"+b]=ev[a]*ev[b]
interaction_cols=[c for c in ev.columns if c.startswith("I_")]
combined_cols=all_cols+interaction_cols


# ---------- MATHEMATICAL / STATISTICAL MODEL STUDY ----------
# Keep the event definition and train/cal/test split identical to Macro x Micro v3
# so that differences come from the mathematical/statistical model, not from relabelling.

def wilson(k,n,z=1.6448536269514722):
    if n<=0:return 0.0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

train=ev[(ev.year>=2015)&(ev.year<=2018)&(ev.tradable==1)].copy()
cal=ev[(ev.year==2019)&(ev.tradable==1)].copy()
test=ev[(ev.year>=2020)&(ev.year<=2026)&(ev.tradable==1)].copy()
print("EVENTS",len(ev),"SPLITS",len(train),len(cal),len(test),flush=True)

# De-duplicate feature names defensively.
combined_cols=list(dict.fromkeys(combined_cols))

# Reduced, economically/statistically interpretable set for quadratic/spline models.
preferred=[
 "M_breadth5","M_breadth20","M_breadth_accel","M_eqret5","M_eqret20",
 "M_lucount5","M_lu_accel","M_up5_5","M_risk_dispersion","M_mkt_amt_ratio20",
 "S_ret5","S_ret20","S_ret60","S_accel5","S_rv20","S_high20_dist",
 "S_ma20_dist","S_prior_lu20","S_marcap_log",
 "X_gap0","X_low_ret0","X_range0","X_one_price","X_vol_ratio20",
 "X_amt_ratio20","X_turnover","X_amount_log","d1_gap"
]
reduced=[x for x in preferred if x in ev.columns]

def logit_l2():
    return Pipeline([
        ("s",StandardScaler()),
        ("m",LogisticRegression(C=.20,max_iter=7000,class_weight="balanced",random_state=SEED))
    ])

def logit_elastic():
    return Pipeline([
        ("s",StandardScaler()),
        ("m",LogisticRegression(
            C=.18,penalty="elasticnet",solver="saga",l1_ratio=.25,
            max_iter=10000,class_weight="balanced",random_state=SEED))
    ])

def lda_shrink():
    return Pipeline([
        ("s",StandardScaler()),
        ("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))
    ])

def qda_reg():
    return Pipeline([
        ("s",StandardScaler()),
        ("m",QuadraticDiscriminantAnalysis(reg_param=.55))
    ])

def spline_logit():
    return Pipeline([
        ("sp",SplineTransformer(n_knots=4,degree=2,include_bias=False)),
        ("s",StandardScaler()),
        ("m",LogisticRegression(C=.12,max_iter=7000,class_weight="balanced",random_state=SEED))
    ])

def copula_logit(n):
    return Pipeline([
        ("q",QuantileTransformer(n_quantiles=min(160,n),output_distribution="normal",random_state=SEED)),
        ("m",LogisticRegression(C=.20,max_iter=7000,class_weight="balanced",random_state=SEED))
    ])

def metrics_full(y,score):
    return {
      "pr_auc":round(float(average_precision_score(y,score)),4),
      "roc_auc":round(float(roc_auc_score(y,score)),4),
      "brier":round(float(brier_score_loss(y,np.clip(score,0,1))),4)
    }

def choose_threshold(cp):
    opts=[]
    for q in [0.70,0.75,0.80,0.85,0.90,0.925,0.95]:
        th=float(np.quantile(cp,q))
        s=cal[cp>=th]; n=len(s); k=int(s.positive.sum())
        if n>=12:
            opts.append((wilson(k,n),k/n,n,q,th,k))
    return max(opts,key=lambda z:(z[0],z[1],z[2]))

base_rate=float(test.positive.mean())

def summarize_selection(name,score_cal,score_test,model_kind,features):
    lb_cal,prec_cal,n_cal,q,th,k_cal=choose_threshold(score_cal)
    mask=score_test>=th
    s=test[mask].copy(); N=len(s); K=int(s.positive.sum())
    # Fixed top-decile result is not outcome-optimized; it is a robustness check.
    th90=float(np.quantile(score_cal,.90))
    s90=test[score_test>=th90].copy(); N90=len(s90); K90=int(s90.positive.sum())

    a,b=1+K,1+N-K
    ci_lo=float(beta_dist.ppf(.05,a,b)) if N else np.nan
    ci_hi=float(beta_dist.ppf(.95,a,b)) if N else np.nan
    pval=float(binomtest(K,N,p=base_rate,alternative="greater").pvalue) if N else 1.0

    yearly=[]
    rates=[]
    for y in range(2020,2027):
        z=s[s.year==y]; nn=len(z); kk=int(z.positive.sum())
        rr=kk/nn if nn else np.nan
        if nn: rates.append(rr)
        yearly.append({"year":y,"n":nn,"k":kk,"rate":round(100*rr,2) if nn else None})

    return {
      "model":name,"kind":model_kind,"features":features,
      "cal":{"q":q,"n":n_cal,"k":k_cal,"rate":round(100*prec_cal,2),"wilson90_lb":round(100*lb_cal,2)},
      "test":{
        "n":N,"k":K,"rate":round(100*K/N,2) if N else None,
        "wilson90_lb":round(100*wilson(K,N),2) if N else None,
        "beta90_ci":[round(100*ci_lo,2),round(100*ci_hi,2)] if N else None,
        "lift_vs_base":round((K/N)/base_rate,3) if N and base_rate>0 else None,
        "p_vs_base":pval,
        "continuation_precision":round(100*s.continuation.mean(),2) if N else None,
        "failure_rate":round(100*s.failure.mean(),2) if N else None,
        "avg_mfe":round(float(s.mfe.mean()),2) if N else None,
        "avg_mae":round(float(s.mae.mean()),2) if N else None,
        "avg_d5":round(float(s.d5ret.mean()),2) if N else None,
        "year_rate_sd":round(100*float(np.std(rates,ddof=0)),2) if rates else None
      },
      "fixed_top10":{
        "n":N90,"k":K90,"rate":round(100*K90/N90,2) if N90 else None,
        "wilson90_lb":round(100*wilson(K90,N90),2) if N90 else None
      },
      "full_test":metrics_full(test.positive,score_test),
      "yearly":yearly
    }

scores_cal={}
scores_test={}
results=[]

# 1) Regularized logistic regression: linear log-odds / maximum likelihood baseline.
m=logit_l2(); m.fit(train[combined_cols],train.positive)
scores_cal["L2_LOGIT"]=m.predict_proba(cal[combined_cols])[:,1]
scores_test["L2_LOGIT"]=m.predict_proba(test[combined_cols])[:,1]
results.append(summarize_selection("L2_LOGIT",scores_cal["L2_LOGIT"],scores_test["L2_LOGIT"],
                                   "regularized logistic likelihood",len(combined_cols)))

# 2) Elastic-net logistic: sparse + ridge shrinkage.
m=logit_elastic(); m.fit(train[combined_cols],train.positive)
scores_cal["ELASTIC_LOGIT"]=m.predict_proba(cal[combined_cols])[:,1]
scores_test["ELASTIC_LOGIT"]=m.predict_proba(test[combined_cols])[:,1]
results.append(summarize_selection("ELASTIC_LOGIT",scores_cal["ELASTIC_LOGIT"],scores_test["ELASTIC_LOGIT"],
                                   "elastic-net logistic likelihood",len(combined_cols)))

# 3) Shrinkage LDA: shared covariance Gaussian discrimination.
m=lda_shrink(); m.fit(train[combined_cols],train.positive)
scores_cal["LDA_SHRINK"]=m.predict_proba(cal[combined_cols])[:,1]
scores_test["LDA_SHRINK"]=m.predict_proba(test[combined_cols])[:,1]
results.append(summarize_selection("LDA_SHRINK",scores_cal["LDA_SHRINK"],scores_test["LDA_SHRINK"],
                                   "shrinkage discriminant analysis",len(combined_cols)))

# 4) Regularized QDA: allows quadratic class boundaries, on reduced features to control variance.
m=qda_reg(); m.fit(train[reduced],train.positive)
scores_cal["QDA_REG"]=m.predict_proba(cal[reduced])[:,1]
scores_test["QDA_REG"]=m.predict_proba(test[reduced])[:,1]
results.append(summarize_selection("QDA_REG",scores_cal["QDA_REG"],scores_test["QDA_REG"],
                                   "regularized quadratic discriminant",len(reduced)))

# 5) Spline logistic: GAM-like nonlinear additive response without tree splits.
m=spline_logit(); m.fit(train[reduced],train.positive)
scores_cal["SPLINE_LOGIT"]=m.predict_proba(cal[reduced])[:,1]
scores_test["SPLINE_LOGIT"]=m.predict_proba(test[reduced])[:,1]
results.append(summarize_selection("SPLINE_LOGIT",scores_cal["SPLINE_LOGIT"],scores_test["SPLINE_LOGIT"],
                                   "spline generalized additive logit",len(reduced)))

# 6) Gaussian-copula logistic: rank-normalize marginals, then linear dependence model.
m=copula_logit(len(train)); m.fit(train[combined_cols],train.positive)
scores_cal["COPULA_LOGIT"]=m.predict_proba(cal[combined_cols])[:,1]
scores_test["COPULA_LOGIT"]=m.predict_proba(test[combined_cols])[:,1]
results.append(summarize_selection("COPULA_LOGIT",scores_cal["COPULA_LOGIT"],scores_test["COPULA_LOGIT"],
                                   "semiparametric Gaussian-copula logit",len(combined_cols)))

# 7) Hurdle / factorized probability:
#    P(trade success) = P(continuation) * P(success | continuation).
mc=logit_l2(); mc.fit(train[combined_cols],train.continuation)
trc=train[train.continuation==1].copy()
# positive among continuation is exactly TP-before-close-stop under the current trade label.
me=logit_l2(); me.fit(trc[combined_cols],trc.positive)
scores_cal["HURDLE_FACTOR"]=(mc.predict_proba(cal[combined_cols])[:,1]*
                            me.predict_proba(cal[combined_cols])[:,1])
scores_test["HURDLE_FACTOR"]=(mc.predict_proba(test[combined_cols])[:,1]*
                              me.predict_proba(test[combined_cols])[:,1])
results.append(summarize_selection("HURDLE_FACTOR",scores_cal["HURDLE_FACTOR"],scores_test["HURDLE_FACTOR"],
                                   "factorized conditional probability",len(combined_cols)))

# 8) Competing-risk algebra:
#    continuation probability is penalized by failure probability.
mf=logit_l2(); mf.fit(train[combined_cols],train.failure)
scores_cal["COMPETING_LOGIT"]=(mc.predict_proba(cal[combined_cols])[:,1]*
                               (1-mf.predict_proba(cal[combined_cols])[:,1])*
                               me.predict_proba(cal[combined_cols])[:,1])
scores_test["COMPETING_LOGIT"]=(mc.predict_proba(test[combined_cols])[:,1]*
                                (1-mf.predict_proba(test[combined_cols])[:,1])*
                                me.predict_proba(test[combined_cols])[:,1])
results.append(summarize_selection("COMPETING_LOGIT",scores_cal["COMPETING_LOGIT"],scores_test["COMPETING_LOGIT"],
                                   "competing-risk probability factorization",len(combined_cols)))

# 9) Robust mathematical ensemble using only pre-test scores.
# Convert each model score to an empirical percentile rank on 2019, then geometric mean.
ensemble_names=["L2_LOGIT","ELASTIC_LOGIT","LDA_SHRINK","QDA_REG","SPLINE_LOGIT","COPULA_LOGIT","HURDLE_FACTOR","COMPETING_LOGIT"]
def percentile_from_cal(cal_score,x):
    s=np.sort(cal_score)
    return (np.searchsorted(s,x,side="right")+.5)/(len(s)+1.0)

cal_rank=[]
test_rank=[]
for nm in ensemble_names:
    cp=scores_cal[nm]; tp=scores_test[nm]
    cal_rank.append(np.array([percentile_from_cal(cp,v) for v in cp]))
    test_rank.append(np.array([percentile_from_cal(cp,v) for v in tp]))
cal_rank=np.vstack(cal_rank); test_rank=np.vstack(test_rank)
scores_cal["GEOMETRIC_RANK_ENSEMBLE"]=np.exp(np.mean(np.log(np.clip(cal_rank,1e-6,1)),axis=0))
scores_test["GEOMETRIC_RANK_ENSEMBLE"]=np.exp(np.mean(np.log(np.clip(test_rank,1e-6,1)),axis=0))
results.append(summarize_selection("GEOMETRIC_RANK_ENSEMBLE",
                                   scores_cal["GEOMETRIC_RANK_ENSEMBLE"],
                                   scores_test["GEOMETRIC_RANK_ENSEMBLE"],
                                   "geometric consensus of statistical ranks",len(ensemble_names)))

# 10) Distribution-free conformal selective classifier.
# Fixed alpha=.10; no target-dependent threshold tuning on the test set.
base_cal=scores_cal["L2_LOGIT"]; base_test=scores_test["L2_LOGIT"]
nc0=base_cal[cal.positive.to_numpy()==0]         # nonconformity for class 0 = p(class1)
nc1=(1-base_cal)[cal.positive.to_numpy()==1]     # nonconformity for class 1 = 1-p(class1)
def conformal_pvals(p):
    a0=p
    a1=1-p
    pv0=(1+np.sum(nc0>=a0))/(len(nc0)+1)
    pv1=(1+np.sum(nc1>=a1))/(len(nc1)+1)
    return pv0,pv1
conf_mask=[]
for p in base_test:
    pv0,pv1=conformal_pvals(p)
    conf_mask.append((pv1>.10) and (pv0<=.10))
conf_mask=np.array(conf_mask,bool)
cs=test[conf_mask].copy(); CN=len(cs); CK=int(cs.positive.sum())
conformal={
  "model":"CONFORMAL_SINGLETON_POSITIVE","alpha":0.10,"n":CN,"k":CK,
  "rate":round(100*CK/CN,2) if CN else None,
  "wilson90_lb":round(100*wilson(CK,CN),2) if CN else None,
  "continuation_precision":round(100*cs.continuation.mean(),2) if CN else None,
  "avg_d5":round(float(cs.d5ret.mean()),2) if CN else None
}

# Holm multiplicity correction for the model-vs-base one-sided binomial tests.
ps=[r["test"]["p_vs_base"] for r in results]
order=np.argsort(ps)
adj=[None]*len(ps)
running=0.0
mtests=len(ps)
for rank,idx in enumerate(order):
    val=min(1.0,(mtests-rank)*ps[idx])
    running=max(running,val)
    adj[idx]=running
for r,a in zip(results,adj):
    r["test"]["holm_p_vs_base"]=round(float(a),6)

# Ranking criteria:
# A) user's hard target: precision >=80% with at least 100 OOS signals.
# B) otherwise, highest OOS precision among N>=100, with Wilson LB as tie-break.
eligible=[r for r in results if r["test"]["n"]>=100]
eligible_sorted=sorted(
    eligible,
    key=lambda r:(r["test"]["rate"] if r["test"]["rate"] is not None else -1,
                  r["test"]["wilson90_lb"] if r["test"]["wilson90_lb"] is not None else -1),
    reverse=True
)
best_100=eligible_sorted[0] if eligible_sorted else None
all_sorted=sorted(results,key=lambda r:(r["test"]["rate"] or -1),reverse=True)
best_any=all_sorted[0] if all_sorted else None
pass80=any(r["test"]["n"]>=100 and (r["test"]["rate"] or 0)>=80 for r in results)

result={
 "study":"SANGTTA MATHEMATICAL & STATISTICAL MODEL RESEARCH v1",
 "data":"KOSDAQ actual daily marcap data; same event construction as Macro x Micro v3",
 "split":{"train":"2015-2018","calibration":"2019","untouched_test":"2020-2026"},
 "goal":{"precision_pct":80,"min_oos_signals":100,"entry":"D+1 gap -5%..+3%","tp":"+6%","close_stop":"-3%","horizon":"D+5"},
 "events":{"all":len(ev),"train":len(train),"cal":len(cal),"test":len(test),
           "test_base_positive_rate_pct":round(100*base_rate,2)},
 "models":results,
 "conformal":conformal,
 "best_n_ge_100":best_100,
 "best_any_n":best_any,
 "pass_80_n100":pass80,
 "reference_macro_micro_v3":{"rate_pct":53.57,"n":28,"k":15,"wilson90_lb_pct":38.45}
}
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
open("math_stats_v1_result.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
