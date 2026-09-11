import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
d["illiq_raw"]=(d.ChangesRatio.abs()/100)/(d.Amount/1e8)

# ---------------- Macro / economic state ----------------
g=d.groupby("Date",sort=True)
macro=g.agg(
    breadth=("ChangesRatio",lambda s:float((s>0).mean())),
    up3=("ChangesRatio",lambda s:float((s>=3).mean())),
    up5=("ChangesRatio",lambda s:float((s>=5).mean())),
    dn3=("ChangesRatio",lambda s:float((s<=-3).mean())),
    medret=("ChangesRatio","median"),
    eqret=("ChangesRatio","mean"),
    mkt_amount=("Amount","sum"),
    mkt_illiq=("illiq_raw","median"),
    lucount=("is_lu","sum"),
)
for w in [5,10,20,60]:
    macro[f"breadth{w}"]=macro.breadth.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"up3_{w}"]=macro.up3.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"up5_{w}"]=macro.up5.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"dn3_{w}"]=macro.dn3.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"eqret{w}"]=macro.eqret.rolling(w,min_periods=max(3,w//2)).mean()
    macro[f"amount{w}"]=macro.mkt_amount.rolling(w,min_periods=max(3,w//2)).median()
    macro[f"illiq{w}"]=macro.mkt_illiq.rolling(w,min_periods=max(3,w//2)).median()
    macro[f"lucount{w}"]=macro.lucount.rolling(w,min_periods=max(3,w//2)).mean()
macro["breadth_accel"]=macro.breadth5-macro.breadth20
macro["diffusion_accel"]=macro.up3_5-macro.up3_20
macro["risk_dispersion"]=macro.up5_5+macro.dn3_5
macro["amount_shock"]=macro.mkt_amount/macro.amount20.replace(0,np.nan)
macro["illiq_shock"]=macro.mkt_illiq/macro.illiq20.replace(0,np.nan)
macro["lu_accel"]=macro.lucount5-macro.lucount20
# Smooth macro risk-on score, trained without test labels; same-day D0 close variables only.
macro["risk_on_raw"]=(macro.eqret5.rank(pct=True)+macro.breadth5.rank(pct=True)+macro.up3_5.rank(pct=True)
                      + (1-macro.illiq_shock.rank(pct=True)) + macro.amount_shock.rank(pct=True))/5.0
M=macro.to_dict("index")

d["mcap_pct"]=d.groupby("Date")["Marcap"].rank(pct=True,ascending=False)
event_codes=set(d.loc[d.is_lu,"Code"])
ed=d[d.Code.isin(event_codes)].sort_values(["Code","Date"]).reset_index(drop=True)

TARGETS=[("T3",3.0,5),("T4",4.0,7),("T5",5.0,10)]
events=[]

for code,gg in ed.groupby("Code",sort=False):
    gg=gg.reset_index(drop=True)
    lu=gg.is_lu.to_numpy(bool); inds=np.flatnonzero(lu)
    op=gg.Open.to_numpy(float); hi=gg.High.to_numpy(float); lo=gg.Low.to_numpy(float); cl=gg.Close.to_numpy(float)
    vo=gg.Volume.to_numpy(float); am=gg.Amount.to_numpy(float); mc=gg.Marcap.to_numpy(float); st=gg.Stocks.to_numpy(float)
    ill=gg.illiq_raw.to_numpy(float); dt=gg.Date.to_numpy(); nm=gg.Name.astype(str).to_numpy(); mp=gg.mcap_pct.to_numpy(float)
    last=-999
    for i in inds:
        if i-last<=4: continue
        last=i
        if i<120 or i+15>=len(gg): continue
        date=pd.Timestamp(dt[i]); mm=M.get(date)
        if not mm: continue

        # Entry: first D+1..D+3 open within +/-3% of D0 close.
        # If a prior close breaks -3% vs D0 close, later entry is cancelled.
        e=None; broken=False
        for j in range(i+1,i+4):
            if j>i+1 and cl[j-1] <= cl[i]*0.97: broken=True
            if broken: break
            gap=(op[j]/cl[i]-1)*100
            if -3.0 <= gap <= 3.0:
                e=j; break
        if e is None:
            continue

        entry=op[e]
        entry_day=e-i
        entry_gap=(entry/cl[i]-1)*100
        entry_gap_prev=(entry/cl[e-1]-1)*100
        if e==i+1:
            pre_max=pre_min=pre_last=pre_minclose=0.0
        else:
            p=np.arange(i+1,e)
            pre_max=(np.max(hi[p])/cl[i]-1)*100
            pre_min=(np.min(lo[p])/cl[i]-1)*100
            pre_last=(cl[e-1]/cl[i]-1)*100
            pre_minclose=(np.min(cl[p])/cl[i]-1)*100

        labels={}
        pnl={}
        for name,tpct,horizon in TARGETS:
            end=min(e+horizon-1,len(gg)-1)
            success=0; gross=None; reason="TIME"
            for j in range(e,end+1):
                # target is intraday; stop is close-based, so target can execute before close-stop same day.
                if hi[j] >= entry*(1+tpct/100):
                    success=1; gross=tpct; reason="TP"; break
                if cl[j] <= entry*0.97:
                    success=0; gross=(cl[j]/entry-1)*100; reason="STOP_CLOSE"; break
            if gross is None:
                gross=(cl[end]/entry-1)*100
            labels[name]=success
            pnl[name]=gross-0.30

        # Early safety / hurdle labels, known only for training targets.
        end3=min(e+2,len(gg)-1)
        safe3=int(np.min(cl[e:end3+1])>entry*0.97)
        up3_any=int(np.max(hi[e:min(e+5,len(gg))])>=entry*1.03)

        # -------- features available by entry open --------
        def ret(n): return (cl[i]/cl[i-n]-1)*100
        r5,r10,r20,r60,r120=[ret(n) for n in [5,10,20,60,120]]
        pre1=(cl[i-1]/cl[i-2]-1)*100; pre5=(cl[i-1]/cl[i-6]-1)*100; pre20=(cl[i-1]/cl[i-21]-1)*100
        rv10=float(np.std((cl[i-9:i+1]/cl[i-10:i]-1)*100,ddof=1))
        rv20=float(np.std((cl[i-19:i+1]/cl[i-20:i]-1)*100,ddof=1))
        medv5=float(np.median(vo[i-5:i])); medv20=float(np.median(vo[i-20:i]))
        meda5=float(np.median(am[i-5:i])); meda20=float(np.median(am[i-20:i]))
        ph20=float(np.max(hi[i-20:i])); ph60=float(np.max(hi[i-60:i])); ph120=float(np.max(hi[i-120:i]))
        ma20=float(np.mean(cl[i-20:i])); ma60=float(np.mean(cl[i-60:i]))
        turn20=float(np.median(vo[i-20:i]/st[i-20:i]))
        amihud20=float(np.median(ill[i-20:i])); amihud60=float(np.median(ill[i-60:i]))
        illiq_shock=ill[i]/amihud20 if amihud20>0 else np.nan

        rec={"Code":code,"Name":nm[i],"Date":date,"year":date.year,
             "entry_day":entry_day,"entry_gap":entry_gap,"entry_gap_prev":entry_gap_prev,
             "preentry_max":pre_max,"preentry_min":pre_min,"preentry_last":pre_last,"preentry_minclose":pre_minclose,
             "safe3":safe3,"up3_any":up3_any}
        rec.update(labels)
        rec.update({f"pnl_{k}":float(v) for k,v in pnl.items()})
        rec.update({
          "mcap_log":math.log1p(mc[i]),"shares_log":math.log1p(st[i]),"mcap_pct":mp[i],
          "turnover_shares":vo[i]/st[i],"turnover_shares_ratio20":(vo[i]/st[i])/turn20 if turn20>0 else np.nan,
          "amount_to_mcap":am[i]/mc[i],
          "amihud20":amihud20,"amihud60":amihud60,"illiq_shock":illiq_shock,
          "pre_ret1":pre1,"pre_ret5":pre5,"pre_ret20":pre20,
          "ret5":r5,"ret10":r10,"ret20":r20,"ret60":r60,"ret120":r120,
          "accel5":r5-r20/4.0,"accel10":r10-r60/6.0,"rv10":rv10,"rv20":rv20,
          "high20_dist":cl[i]/ph20-1,"high60_dist":cl[i]/ph60-1,"high120_dist":cl[i]/ph120-1,
          "ma20_dist":cl[i]/ma20-1,"ma60_dist":cl[i]/ma60-1,
          "prior_lu20":float(np.sum(lu[i-20:i])),"prior_lu60":float(np.sum(lu[i-60:i])),"prior_lu120":float(np.sum(lu[i-120:i])),
          "gap0":(op[i]/cl[i-1]-1)*100,"low_ret0":(lo[i]/cl[i-1]-1)*100,
          "range0":(hi[i]/lo[i]-1)*100,"open_to_close0":(cl[i]/op[i]-1)*100,
          "one_price":float(op[i]==hi[i]==lo[i]==cl[i]),
          "volume_ratio5":vo[i]/medv5 if medv5>0 else np.nan,"volume_ratio20":vo[i]/medv20 if medv20>0 else np.nan,
          "amount_ratio5":am[i]/meda5 if meda5>0 else np.nan,"amount_ratio20":am[i]/meda20 if meda20>0 else np.nan,
          "amount_log":math.log1p(am[i]),
          "M_breadth":mm["breadth"],"M_breadth5":mm["breadth5"],"M_breadth20":mm["breadth20"],
          "M_breadth_accel":mm["breadth_accel"],"M_up3_5":mm["up3_5"],"M_up5_5":mm["up5_5"],
          "M_diffusion_accel":mm["diffusion_accel"],"M_risk_dispersion":mm["risk_dispersion"],
          "M_eqret5":mm["eqret5"],"M_eqret20":mm["eqret20"],"M_amount_shock":mm["amount_shock"],
          "M_illiq_shock":mm["illiq_shock"],"M_lucount":mm["lucount"],"M_lucount5":mm["lucount5"],
          "M_lu_accel":mm["lu_accel"],"M_risk_on":mm["risk_on_raw"],
        })
        if all(np.isfinite(v) for k,v in rec.items() if isinstance(v,(int,float,np.integer,np.floating))):
            events.append(rec)

ev=pd.DataFrame(events).sort_values("Date").reset_index(drop=True)
NONFEAT=["Code","Name","Date","year","safe3","up3_any"]+[x[0] for x in TARGETS]+[f"pnl_{x[0]}" for x in TARGETS]
features=[c for c in ev.columns if c not in NONFEAT]

train=ev[(ev.year>=2015)&(ev.year<=2018)].copy()
cal=ev[ev.year==2019].copy()
test=ev[(ev.year>=2020)&(ev.year<=2026)].copy()
print("SPLITS",len(train),len(cal),len(test),flush=True)

def hgb():
    return HistGradientBoostingClassifier(max_depth=3,learning_rate=.03,max_iter=260,l2_regularization=8,min_samples_leaf=18,random_state=SEED)
def elastic():
    return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.15,penalty="elasticnet",solver="saga",l1_ratio=.20,
                        class_weight="balanced",max_iter=10000,random_state=SEED))])
def lda():
    return Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])
def wilson(k,n,z=1.6448536269514722):
    if n<=0:return 0.0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

# Walk-forward OOF base predictions for meta model, no future leakage.
def oof_meta(target):
    parts=[]
    for vy in [2016,2017,2018]:
        tr=ev[(ev.year>=2015)&(ev.year<vy)].copy()
        va=ev[ev.year==vy].copy()
        if len(tr)<200 or len(va)==0: continue
        mods=[hgb(),elastic(),lda()]
        preds=[]
        for m in mods:
            m.fit(tr[features],tr[target]); preds.append(m.predict_proba(va[features])[:,1])
        # Hurdle heads: safety and target-if-safe
        ms=hgb(); ms.fit(tr[features],tr.safe3)
        safe_tr=tr[tr.safe3==1]
        mt=hgb(); mt.fit(safe_tr[features],safe_tr[target])
        ps=ms.predict_proba(va[features])[:,1]
        pt=mt.predict_proba(va[features])[:,1]
        z=va[["Date","year",target]].copy()
        z["p_hgb"],z["p_elastic"],z["p_lda"]=preds
        z["p_hurdle"]=ps*pt
        z["M_risk_on"]=va["M_risk_on"].to_numpy()
        z["M_illiq_shock"]=va["M_illiq_shock"].to_numpy()
        z["entry_gap"]=va["entry_gap"].to_numpy()
        z["turnover_shares_ratio20"]=va["turnover_shares_ratio20"].to_numpy()
        parts.append(z)
    return pd.concat(parts,ignore_index=True)

def fit_final_scores(target):
    # base models fit through 2018
    mods=[hgb(),elastic(),lda()]
    cplist=[]; tplist=[]
    for m in mods:
        m.fit(train[features],train[target])
        cplist.append(m.predict_proba(cal[features])[:,1])
        tplist.append(m.predict_proba(test[features])[:,1])
    ms=hgb(); ms.fit(train[features],train.safe3)
    safe_tr=train[train.safe3==1]
    mt=hgb(); mt.fit(safe_tr[features],safe_tr[target])
    cph=ms.predict_proba(cal[features])[:,1]*mt.predict_proba(cal[features])[:,1]
    tph=ms.predict_proba(test[features])[:,1]*mt.predict_proba(test[features])[:,1]

    oof=oof_meta(target)
    meta_cols=["p_hgb","p_elastic","p_lda","p_hurdle","M_risk_on","M_illiq_shock","entry_gap","turnover_shares_ratio20"]
    meta=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.2,class_weight="balanced",max_iter=7000,random_state=SEED))])
    meta.fit(oof[meta_cols],oof[target])

    cmeta=pd.DataFrame({"p_hgb":cplist[0],"p_elastic":cplist[1],"p_lda":cplist[2],"p_hurdle":cph,
                        "M_risk_on":cal.M_risk_on.values,"M_illiq_shock":cal.M_illiq_shock.values,
                        "entry_gap":cal.entry_gap.values,"turnover_shares_ratio20":cal.turnover_shares_ratio20.values})
    tmeta=pd.DataFrame({"p_hgb":tplist[0],"p_elastic":tplist[1],"p_lda":tplist[2],"p_hurdle":tph,
                        "M_risk_on":test.M_risk_on.values,"M_illiq_shock":test.M_illiq_shock.values,
                        "entry_gap":test.entry_gap.values,"turnover_shares_ratio20":test.turnover_shares_ratio20.values})
    cp=meta.predict_proba(cmeta)[:,1]
    tp=meta.predict_proba(tmeta)[:,1]
    return cp,tp,meta

def choose_precision_threshold(cp,target):
    opts=[]
    # precision-first. Threshold is selected ONLY on 2019.
    for q in [0.50,0.60,0.70,0.75,0.80,0.85,0.875,0.90,0.925,0.95]:
        th=float(np.quantile(cp,q))
        s=cal[cp>=th]; n=len(s); k=int(s[target].sum())
        if n>=15:
            opts.append((k/n,wilson(k,n),n,q,th,k))
    # hard ordering: precision, then Wilson LB, then N
    return max(opts,key=lambda x:(x[0],x[1],x[2]))

def evaluate(target,cp,tp):
    p,lb,n,q,th,k=choose_precision_threshold(cp,target)
    s=test[tp>=th].copy(); N=len(s); K=int(s[target].sum())
    yearly=[]
    for y in range(2020,2027):
        z=s[s.year==y]; nn=len(z); kk=int(z[target].sum())
        yearly.append({"year":y,"n":nn,"k":kk,"rate":round(100*kk/nn,2) if nn else None,
                       "avg_net":round(float(z[f"pnl_{target}"].mean()),3) if nn else None})
    return {
      "target":target,
      "cal":{"q":q,"n":n,"k":k,"rate":round(100*p,2),"wilson90_lb":round(100*lb,2)},
      "test":{"n":N,"k":K,"rate":round(100*K/N,2) if N else None,
              "wilson90_lb":round(100*wilson(K,N),2) if N else None,
              "avg_net_pct":round(float(s[f"pnl_{target}"].mean()),3) if N else None,
              "median_net_pct":round(float(s[f"pnl_{target}"].median()),3) if N else None,
              "positive_net_pct":round(100*float((s[f"pnl_{target}"]>0).mean()),2) if N else None,
              "risk_on_avg":round(float(s.M_risk_on.mean()),3) if N else None,
              "entry_gap_avg":round(float(s.entry_gap.mean()),3) if N else None},
      "yearly":yearly,
      "threshold":th
    }

all_results=[]
scores={}
for target,_,_ in TARGETS:
    cp,tp,_=fit_final_scores(target)
    res=evaluate(target,cp,tp)
    all_results.append(res)
    scores[target]=(cp,tp)

# Development-choice target: choose by 2019 precision first, then Wilson lower bound.
chosen=max(all_results,key=lambda r:(r["cal"]["rate"],r["cal"]["wilson90_lb"],r["cal"]["n"]))
chosen_target=chosen["target"]
cp,tp=scores[chosen_target]

# Conformal-style abstention on the chosen target using 2019 calibration residuals.
# A positive is emitted only if positive class remains plausible and negative class is rejected at alpha.
ycal=cal[chosen_target].to_numpy()
nc0=cp[ycal==0]
nc1=1-cp[ycal==1]
def singleton_mask(alpha):
    mask=[]
    for p in tp:
        pv0=(1+np.sum(nc0>=p))/(len(nc0)+1)
        pv1=(1+np.sum(nc1>=(1-p)))/(len(nc1)+1)
        mask.append((pv1>alpha) and (pv0<=alpha))
    return np.array(mask,bool)
conformal=[]
for alpha in [0.05,0.10,0.15,0.20]:
    m=singleton_mask(alpha); s=test[m]; N=len(s); K=int(s[chosen_target].sum())
    conformal.append({"alpha":alpha,"n":N,"k":K,"rate":round(100*K/N,2) if N else None,
                      "wilson90_lb":round(100*wilson(K,N),2) if N else None,
                      "avg_net_pct":round(float(s[f"pnl_{chosen_target}"].mean()),3) if N else None})

# Fixed selectivity audit: no test optimization.
fixed=[]
for q in [.80,.85,.90,.925,.95]:
    th=float(np.quantile(cp,q)); s=test[tp>=th]; N=len(s); K=int(s[chosen_target].sum())
    fixed.append({"q":q,"n":N,"k":K,"rate":round(100*K/N,2) if N else None,
                  "wilson90_lb":round(100*wilson(K,N),2) if N else None,
                  "avg_net_pct":round(float(s[f"pnl_{chosen_target}"].mean()),3) if N else None})

result={
 "study":"KQ Precision-First Regime Hurdle v1",
 "principles":[
   "precision-first rather than return-maximization",
   "regularized statistics + nonlinear model + shrinkage LDA",
   "economic liquidity/regime state",
   "hurdle factorization for early safety and target reach",
   "walk-forward OOF meta model",
   "2019-only threshold selection and conformal abstention"
 ],
 "entry_rule":"first D+1..D+3 open within -3%..+3% of first-limit-up close; prior -3% close break cancels later entry",
 "stop":"-3% close",
 "cost":"0.30% round-trip estimate",
 "candidate_targets":{"T3":"+3% within 5d","T4":"+4% within 7d","T5":"+5% within 10d"},
 "split":{"train":"2015-2018","calibration":"2019","historical_audit":"2020-2026"},
 "note":"2020-2026 is a historical audit, not pristine OOS anymore, because prior model research has already examined this era.",
 "events":{"train":len(train),"cal":len(cal),"audit":len(test)},
 "base_rates_audit_pct":{t:round(100*float(test[t].mean()),2) for t,_,_ in TARGETS},
 "target_results":all_results,
 "chosen_target_by_2019":chosen_target,
 "chosen_result":chosen,
 "conformal_audit":conformal,
 "fixed_selectivity_audit":fixed,
 "promotion_rule":{"minimum_signals":100,"target_win_rate_pct":75,"minimum_avg_net_pct":0},
 "pass":bool(chosen["test"]["n"]>=100 and chosen["test"]["rate"]>=75 and chosen["test"]["avg_net_pct"]>0)
}
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
open("precision_first_v1_result.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
