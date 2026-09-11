import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

SEED=42
YEARS=range(2015,2027)
COLS=["Date","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Market"]

# ---------------- load actual KOSDAQ daily data ----------------
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
d=d[(d.Open>0)&(d.High>0)&(d.Low>0)&(d.Close>0)&(d.Amount>0)].copy()
d=d[~d["Name"].astype(str).str.contains("스팩",na=False)].copy()
d["is_lu"]=(d["ChangesRatio"]>=29.5)&(d["Close"]==d["High"])
d["illiq_raw"]=(d["ChangesRatio"].abs()/100.0)/(d["Amount"]/1e8) # abs return per KRW 100m turnover

# ---------------- aggregate macro/liquidity state ----------------
g=d.groupby("Date",sort=True)
macro=g.agg(
    eqret=("ChangesRatio","mean"),
    medret=("ChangesRatio","median"),
    breadth=("ChangesRatio",lambda s:float((s>0).mean())),
    up5=("ChangesRatio",lambda s:float((s>=5).mean())),
    dn5=("ChangesRatio",lambda s:float((s<=-5).mean())),
    mkt_amount=("Amount","sum"),
    macro_illiq=("illiq_raw","median"),
    lucount=("is_lu","sum"),
)
for w in [5,20,60]:
    macro[f"eqret{w}"]=macro.eqret.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"breadth{w}"]=macro.breadth.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"illiq{w}"]=macro.macro_illiq.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"amt{w}"]=macro.mkt_amount.rolling(w,min_periods=max(3,w//2)).median()
    macro[f"lucount{w}"]=macro.lucount.rolling(w,min_periods=max(3,w//2)).mean()
macro["liq_shock"]=macro.macro_illiq/(macro.illiq20.replace(0,np.nan))
macro["amount_shock"]=macro.mkt_amount/(macro.amt20.replace(0,np.nan))
macro["breadth_accel"]=macro.breadth5-macro.breadth20
macro["eqret_accel"]=macro.eqret5-macro.eqret20

# Hawkes-style self-excitation proxy from lagged limit-up counts only.
# It is an adapted exponential-kernel intensity, not a full tick-level Hawkes fit.
for decay,name in [(0.35,"hawkes_fast"),(0.70,"hawkes_slow")]:
    vals=[]; state=0.0
    for x in macro.lucount.shift(1).fillna(0).to_numpy(float):
        state=x+decay*state
        vals.append(state)
    macro[name]=vals

# Hamilton-style 2-state causal Markov filter.
hmm_cols=["eqret","breadth","macro_illiq","amount_shock","lucount"]
mtrain=macro.loc[(macro.index.year>=2015)&(macro.index.year<=2018),hmm_cols].dropna()
sc=StandardScaler().fit(mtrain)
Xtr=sc.transform(mtrain)
hmm=GaussianHMM(n_components=2,covariance_type="diag",n_iter=300,random_state=SEED).fit(Xtr)
# risk-on state: higher standardized eqret + breadth - illiquidity
means=hmm.means_
risk_score=means[:,0]+means[:,1]-means[:,2]
risk_state=int(np.argmax(risk_score))

def causal_hmm_filter(X):
    trans=hmm.transmat_; start=hmm.startprob_; means=hmm.means_; cov=hmm.covars_
    if cov.ndim==3:
        cov=np.diagonal(cov,axis1=1,axis2=2)
    prev=start.copy(); out=[]
    for x in X:
        if np.any(~np.isfinite(x)):
            out.append(np.nan); continue
        pred=prev@trans
        # diag gaussian log likelihood
        loglik=-0.5*(np.sum(np.log(2*np.pi*cov),axis=1)+np.sum((x-means)**2/cov,axis=1))
        lp=np.log(np.clip(pred,1e-15,1))+loglik
        post=np.exp(lp-logsumexp(lp))
        out.append(float(post[risk_state]))
        prev=post
    return np.array(out)

mx=macro[hmm_cols].copy()
# Fill missing only from past rolling medians, then training medians (fixed).
for c in hmm_cols:
    mx[c]=mx[c].fillna(mx[c].expanding().median())
    mx[c]=mx[c].fillna(float(mtrain[c].median()))
macro["p_risk_on"]=causal_hmm_filter(sc.transform(mx))
M=macro.to_dict("index")

# ---------------- event features & labels ----------------
event_codes=set(d.loc[d.is_lu,"Code"])
ed=d[d.Code.isin(event_codes)].sort_values(["Code","Date"]).reset_index(drop=True)
events=[]
for code,gg in ed.groupby("Code",sort=False):
    gg=gg.reset_index(drop=True)
    lu=gg.is_lu.to_numpy(bool); inds=np.flatnonzero(lu)
    op=gg.Open.to_numpy(float); hi=gg.High.to_numpy(float); lo=gg.Low.to_numpy(float); cl=gg.Close.to_numpy(float)
    vo=gg.Volume.to_numpy(float); am=gg.Amount.to_numpy(float); mc=gg.Marcap.to_numpy(float)
    rr=gg.ChangesRatio.to_numpy(float); ill=gg.illiq_raw.to_numpy(float)
    dt=gg.Date.to_numpy(); nm=gg.Name.astype(str).to_numpy()
    last=-999
    for i in inds:
        if i-last<=4: continue
        last=i
        if i<120 or i+60>=len(gg): continue
        date=pd.Timestamp(dt[i]); mm=M.get(date)
        if not mm: continue

        # classic economic/microstructure proxies
        amihud20=float(np.mean(ill[i-20:i]))
        amihud60=float(np.mean(ill[i-60:i]))
        illiq_shock=ill[i]/amihud20 if amihud20>0 else np.nan
        turnover=am[i]/mc[i] if mc[i]>0 else np.nan
        turnover20=np.median(am[i-20:i]/mc[i-20:i])
        turnover_shock=turnover/turnover20 if turnover20>0 else np.nan

        ret1=(cl[i-19:i+1]/cl[i-20:i]-1)
        cov=np.cov(ret1[1:],ret1[:-1],ddof=1)[0,1]
        roll_spread_proxy=2*np.sqrt(max(-cov,0.0))

        r5=(cl[i]/cl[i-5]-1)*100; r10=(cl[i]/cl[i-10]-1)*100
        r20=(cl[i]/cl[i-20]-1)*100; r60=(cl[i]/cl[i-60]-1)*100; r120=(cl[i]/cl[i-120]-1)*100
        rv20=float(np.std((cl[i-19:i+1]/cl[i-20:i]-1)*100,ddof=1))
        meda20=float(np.median(am[i-20:i])); medv20=float(np.median(vo[i-20:i]))
        ph20=float(np.max(hi[i-20:i])); ph60=float(np.max(hi[i-60:i]))
        gap0=(op[i]/cl[i-1]-1)*100; low_ret0=(lo[i]/cl[i-1]-1)*100
        range0=(hi[i]/lo[i]-1)*100; one_price=float(op[i]==hi[i]==lo[i]==cl[i])
        d1_gap=(op[i+1]/cl[i]-1)*100

        # own-stock Hawkes-style excitation: prior LU events, exponentially decayed by trading-day distance
        own_fast=0.0; own_slow=0.0
        for j in np.flatnonzero(lu[:i]):
            dist=i-j
            if dist<=120:
                own_fast+=math.exp(-dist/5.0)
                own_slow+=math.exp(-dist/20.0)

        vals={
          "size_log":math.log1p(mc[i]),"amount_log":math.log1p(am[i]),
          "amihud20":amihud20,"amihud60":amihud60,"illiq_shock":illiq_shock,
          "turnover":turnover,"turnover_shock":turnover_shock,"roll_spread_proxy":roll_spread_proxy,
          "ret5":r5,"ret10":r10,"ret20":r20,"ret60":r60,"ret120":r120,
          "mom_accel5":r5-r20/4.0,"mom_accel10":r10-r60/6.0,"rv20":rv20,
          "amount_ratio20":am[i]/meda20 if meda20>0 else np.nan,
          "volume_ratio20":vo[i]/medv20 if medv20>0 else np.nan,
          "high20_dist":cl[i]/ph20-1,"high60_dist":cl[i]/ph60-1,
          "gap0":gap0,"low_ret0":low_ret0,"range0":range0,"one_price":one_price,
          "prior_lu20":float(np.sum(lu[i-20:i])),"prior_lu60":float(np.sum(lu[i-60:i])),
          "own_hawkes_fast":own_fast,"own_hawkes_slow":own_slow,
          "macro_breadth":mm["breadth"],"macro_breadth5":mm["breadth5"],"macro_breadth20":mm["breadth20"],
          "macro_eqret":mm["eqret"],"macro_eqret5":mm["eqret5"],"macro_eqret20":mm["eqret20"],
          "macro_illiq":mm["macro_illiq"],"macro_illiq20":mm["illiq20"],"macro_liq_shock":mm["liq_shock"],
          "macro_amount_shock":mm["amount_shock"],"macro_lucount":mm["lucount"],"macro_lucount5":mm["lucount5"],
          "macro_hawkes_fast":mm["hawkes_fast"],"macro_hawkes_slow":mm["hawkes_slow"],
          "p_risk_on":mm["p_risk_on"],"d1_gap":d1_gap,
        }
        if any(not np.isfinite(v) for v in vals.values()): continue

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
            mfe=(np.max(hi[hz])/entry-1)*100
            mae=(np.min(lo[hz])/entry-1)*100
            d5ret=(cl[i+5]/entry-1)*100
            for j in hz:
                if cl[j]<=entry*.97: tp=False; break
                if hi[j]>=entry*1.06: tp=True; break
        cont=ptype in (1,2)
        positive=bool(tradable and cont and tp)
        rec={"Code":code,"Name":nm[i],"Date":date,"year":date.year,"path_type":ptype,
             "continuation":int(cont),"failure":int(ptype==4),"positive":int(positive),"tradable":int(tradable),
             "mfe":mfe,"mae":mae,"d5ret":d5ret}
        rec.update({k:float(v) for k,v in vals.items()}); events.append(rec)

ev=pd.DataFrame(events).sort_values("Date").reset_index(drop=True)
train=ev[(ev.year>=2015)&(ev.year<=2018)&(ev.tradable==1)].copy()
cal=ev[(ev.year==2019)&(ev.tradable==1)].copy()
test=ev[(ev.year>=2020)&(ev.year<=2026)&(ev.tradable==1)].copy()
print("EVENTS",len(ev),"SPLITS",len(train),len(cal),len(test),flush=True)

LIQ=["size_log","amihud20","amihud60","illiq_shock","turnover","turnover_shock","roll_spread_proxy",
     "macro_illiq","macro_illiq20","macro_liq_shock","macro_amount_shock","d1_gap"]
BEH=["ret5","ret10","ret20","ret60","ret120","mom_accel5","mom_accel10","rv20","amount_ratio20","volume_ratio20",
     "high20_dist","high60_dist","gap0","low_ret0","range0","one_price","prior_lu20","prior_lu60",
     "macro_breadth","macro_breadth5","macro_eqret5","macro_lucount5","d1_gap"]
REG=LIQ+["p_risk_on","macro_breadth5","macro_breadth20","macro_eqret5","macro_eqret20","macro_lucount5",
         "macro_hawkes_fast","macro_hawkes_slow","d1_gap"]
HAWK=BEH+["own_hawkes_fast","own_hawkes_slow","macro_hawkes_fast","macro_hawkes_slow","p_risk_on"]
LIQ=list(dict.fromkeys(LIQ))\nBEH=list(dict.fromkeys(BEH))\nREG=list(dict.fromkeys(REG))\nHAWK=list(dict.fromkeys(HAWK))\nFULL=sorted(set(LIQ+BEH+REG+HAWK))

def wilson(k,n,z=1.6448536269514722):
    if n<=0:return 0.0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def logit(cols,target):
    m=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.25,max_iter=5000,class_weight="balanced",random_state=SEED))])
    m.fit(train[cols],train[target]); return m

def hgb(cols,target):
    m=HistGradientBoostingClassifier(max_depth=3,learning_rate=.035,max_iter=200,l2_regularization=5,min_samples_leaf=12,random_state=SEED)
    m.fit(train[cols],train[target]); return m

def choose_threshold(cal_score,target):
    opts=[]
    # predeclared selectivity grid, fixed using 2019 only
    for q in [0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.925,0.95]:
        th=float(np.quantile(cal_score,q))
        s=cal[cal_score>=th]; n=len(s); k=int(s[target].sum())
        if n>=12: opts.append((wilson(k,n),k/n,n,q,th))
    return max(opts,key=lambda z:(z[0],z[1],z[2]))

def eval_score(name,target,cal_score,test_score):
    lb,prec,n,q,th=choose_threshold(cal_score,target)
    s=test[test_score>=th].copy(); N=len(s); K=int(s[target].sum())
    yearly=[]
    for y in range(2020,2027):
        z=s[s.year==y]; nn=len(z); kk=int(z[target].sum())
        yearly.append({"year":y,"n":nn,"k":kk,"rate":round(100*kk/nn,2) if nn else None})
    return {"model":name,"target":target,
            "cal":{"q":q,"n":n,"k":int(round(prec*n)),"rate":round(prec*100,2),"lb":round(lb*100,2)},
            "test":{"n":N,"k":K,"rate":round(100*K/N,2) if N else None,"lb":round(wilson(K,N)*100,2) if N else None,
                    "avg_mfe":round(float(s.mfe.mean()),2) if N else None,
                    "avg_mae":round(float(s.mae.mean()),2) if N else None,
                    "avg_d5":round(float(s.d5ret.mean()),2) if N else None,
                    "continuation_precision":round(100*s.continuation.mean(),2) if N else None},
            "yearly":yearly},th

results=[]

# 1. Amihud/Kyle/Roll inspired liquidity model
for nm,cols in [("ECON_AMIHUD_LIQ",LIQ),("ECON_HONG_STEIN_BEHAV",BEH),("ECON_HAMILTON_LIQ_REGIME",REG),("ECON_HAWKES_ATTENTION",HAWK)]:
    for target in ["continuation","positive"]:
        m=logit(cols,target)
        cs=m.predict_proba(cal[cols])[:,1]; ts=m.predict_proba(test[cols])[:,1]
        r,_=eval_score(nm,target,cs,ts); results.append(r)

# 2. Competing-risk hybrid: separate continuation/failure/execution heads, economics-first
mc=logit(FULL,"continuation"); mf=logit(FULL,"failure"); mp=logit(FULL,"positive")
def cr_score(z):
    pc=mc.predict_proba(z[FULL])[:,1]
    pf=mf.predict_proba(z[FULL])[:,1]
    pp=mp.predict_proba(z[FULL])[:,1]
    # transformed probability: high continuation, low failure, high execution-success
    return np.clip(pc*(1-pf)*np.sqrt(np.clip(pp,1e-9,1)),1e-9,1)
cs=cr_score(cal); ts=cr_score(test)
r,cr_th=eval_score("ECON_COMPETING_RISK","positive",cs,ts); results.append(r)

# 3. New model: economics feature architecture + nonlinear response, threshold still frozen on 2019.
# This tests whether economic structure needs nonlinear interactions rather than linear coefficients.
mcont=hgb(FULL,"continuation"); mfail=hgb(FULL,"failure"); mpos=hgb(FULL,"positive")
def hybrid_score(z):
    pc=mcont.predict_proba(z[FULL])[:,1]
    pf=mfail.predict_proba(z[FULL])[:,1]
    pp=mpos.predict_proba(z[FULL])[:,1]
    # expected-utility-compatible monotone transform (TP +6 / stop -3 => p > 1/3 break-even);
    # selection threshold itself is calibrated on 2019 for precision.
    return np.clip(pc*(1-pf)*pp,1e-9,1)
hcs=hybrid_score(cal); hts=hybrid_score(test)
r,hyb_th=eval_score("SANGTTA_ECON_HYBRID_V1","positive",hcs,hts); results.append(r)

# utility diagnostics on frozen hybrid signals
hs=test[hts>=hyb_th].copy()
if len(hs):
    # nominal binary payoff +6/-3, before costs
    nominal_ev=(hs.positive*6+(1-hs.positive)*(-3)).mean()
else: nominal_ev=np.nan

# model ranking by OOS positive precision, with >=100-signal criterion separately.
positive_rows=[x for x in results if x["target"]=="positive"]
ranked=sorted(positive_rows,key=lambda x:(x["test"]["rate"] if x["test"]["rate"] is not None else -1),reverse=True)

result={
 "study":"ECONOMIC MODEL RESEARCH FOR SANGTTA",
 "economic_foundations":{
   "Amihud":"daily absolute return / value traded; price-impact/illiquidity proxy",
   "Kyle_Roll":"turnover shock and Roll serial-covariance spread proxy; true Kyle lambda requires signed intraday flow",
   "Hamilton":"2-state causal Markov regime filter on KOSDAQ return/breadth/liquidity/turnover/limit-up count",
   "Hong_Stein":"underreaction vs overreaction features from multi-horizon momentum/acceleration/crowding",
   "Hawkes":"adapted exponentially decayed upper-limit-event intensity; true order-arrival Hawkes requires intraday order events",
   "Competing_Risk":"continuation vs failure vs execution success modeled separately"
 },
 "data":"FinanceData/marcap KOSDAQ actual daily data; SPAC excluded; upper-limit proxy ChangesRatio>=29.5 & Close==High",
 "split":{"train":"2015-2018","calibration":"2019","untouched_test":"2020-2026"},
 "goal":{"overall_precision_target_pct":80,"min_test_signals":100,"entry":"D+1 gap -5%..+3%","TP":"+6%","close_stop":"-3%","horizon":"D+5"},
 "events":{"all":len(ev),"train_tradable":len(train),"cal_tradable":len(cal),"test_tradable":len(test)},
 "results":results,
 "hybrid":{"threshold":hyb_th,"test_signals":len(hs),"nominal_binary_EV_pct_per_trade":round(float(nominal_ev),3) if np.isfinite(nominal_ev) else None},
 "best_positive_model":ranked[0] if ranked else None,
 "pass_80_any":any((x["test"]["n"]>=100 and x["test"]["rate"] is not None and x["test"]["rate"]>=80) for x in positive_rows)
}
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
open("econ_model_result.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
