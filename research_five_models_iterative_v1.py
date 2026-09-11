import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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

TARGETS=[("T3",3.0,5),("T4",4.0,7),("T5",5.0,10),("T6",6.0,12)]
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

# ============================================================
# FIVE-MODEL ITERATIVE RESEARCH
# Goal: >=75% hit, >=1.0% mean net return, N>=100.
# All fitting uses 2015-2018, threshold/gates use 2019 only.
# 2020-2026 is a historical audit only.
# ============================================================

def hgb_c(depth=3,l2=8,leaf=18):
    return HistGradientBoostingClassifier(max_depth=depth,learning_rate=.03,max_iter=260,
        l2_regularization=l2,min_samples_leaf=leaf,random_state=SEED)
def hgb_r(loss="squared_error",quantile=None):
    kw=dict(max_depth=3,learning_rate=.03,max_iter=260,l2_regularization=8,min_samples_leaf=18,random_state=SEED,loss=loss)
    if quantile is not None: kw["quantile"]=quantile
    return HistGradientBoostingRegressor(**kw)
def elastic():
    return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.15,penalty="elasticnet",solver="saga",
        l1_ratio=.20,class_weight="balanced",max_iter=10000,random_state=SEED))])
def lda():
    return Pipeline([("s",StandardScaler()),("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])
def wilson(k,n,z=1.6448536269514722):
    if n<=0:return 0.0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den
def pct_thr(x,q): return float(np.quantile(x,q))

# Tail labels.
for target,_,_ in TARGETS:
    ev["TAIL_"+target]=(ev["pnl_"+target]<=-5.0).astype(int)
train=ev[(ev.year>=2015)&(ev.year<=2018)].copy()
cal=ev[ev.year==2019].copy()
test=ev[(ev.year>=2020)&(ev.year<=2026)].copy()

# Reusable component estimates per target.
components={}
for target,_,_ in TARGETS:
    # Win ensemble.
    pcal=[]; ptest=[]
    for m in [hgb_c(),elastic(),lda()]:
        m.fit(train[features],train[target])
        pcal.append(m.predict_proba(cal[features])[:,1]); ptest.append(m.predict_proba(test[features])[:,1])
    pwin_c=np.mean(np.vstack(pcal),axis=0); pwin_t=np.mean(np.vstack(ptest),axis=0)

    # Tail classifier.
    pc=[]; pt=[]
    for m in [hgb_c(),elastic()]:
        m.fit(train[features],train["TAIL_"+target])
        pc.append(m.predict_proba(cal[features])[:,1]); pt.append(m.predict_proba(test[features])[:,1])
    ptail_c=np.mean(np.vstack(pc),axis=0); ptail_t=np.mean(np.vstack(pt),axis=0)

    # Early-safety hurdle.
    ms=hgb_c(); ms.fit(train[features],train.safe3)
    ts=train[train.safe3==1]
    mt=hgb_c(); mt.fit(ts[features],ts[target])
    ph_c=ms.predict_proba(cal[features])[:,1]*mt.predict_proba(cal[features])[:,1]
    ph_t=ms.predict_proba(test[features])[:,1]*mt.predict_proba(test[features])[:,1]

    # Mean and lower-quantile PnL.
    mr=hgb_r(); mr.fit(train[features],train["pnl_"+target])
    mu_c=mr.predict(cal[features]); mu_t=mr.predict(test[features])
    qr=hgb_r(loss="quantile",quantile=.20); qr.fit(train[features],train["pnl_"+target])
    q20_c=qr.predict(cal[features]); q20_t=qr.predict(test[features])

    # Regime-specific experts, using training median only.
    risk_cut=float(train.M_risk_on.median())
    lo=train[train.M_risk_on<risk_cut]; hi=train[train.M_risk_on>=risk_cut]
    ml=hgb_c(depth=2,l2=10,leaf=15); mh=hgb_c(depth=2,l2=10,leaf=15)
    ml.fit(lo[features],lo[target]); mh.fit(hi[features],hi[target])
    preg_c=np.where(cal.M_risk_on.to_numpy()>=risk_cut,
        mh.predict_proba(cal[features])[:,1],ml.predict_proba(cal[features])[:,1])
    preg_t=np.where(test.M_risk_on.to_numpy()>=risk_cut,
        mh.predict_proba(test[features])[:,1],ml.predict_proba(test[features])[:,1])

    components[target]=dict(
        pwin_c=pwin_c,pwin_t=pwin_t,ptail_c=ptail_c,ptail_t=ptail_t,
        ph_c=ph_c,ph_t=ph_t,mu_c=mu_c,mu_t=mu_t,q20_c=q20_c,q20_t=q20_t,
        preg_c=preg_c,preg_t=preg_t,risk_cut=risk_cut)

def eval_mask(target,mask):
    z=test[mask].copy(); n=len(z); k=int(z[target].sum())
    yrs=[]
    for y in range(2020,2027):
        a=z[z.year==y]; nn=len(a); kk=int(a[target].sum())
        yrs.append({"year":y,"n":nn,"rate":round(100*kk/nn,2) if nn else None,
                    "avg_net":round(float(a["pnl_"+target].mean()),3) if nn else None})
    return dict(n=n,k=k,win_rate=100*k/n if n else np.nan,
                lb=100*wilson(k,n) if n else np.nan,
                avg=float(z["pnl_"+target].mean()) if n else np.nan,
                med=float(z["pnl_"+target].median()) if n else np.nan,
                tail=100*float(z["TAIL_"+target].mean()) if n else np.nan,
                yearly=yrs)

def select_threshold(cal_score,target,min_n=15):
    cand=[]
    for q in [.50,.60,.70,.75,.80,.85,.875,.90,.925,.95]:
        th=pct_thr(cal_score,q); m=cal_score>=th; z=cal[m]; n=len(z)
        if n<min_n: continue
        k=int(z[target].sum()); wr=k/n; avg=float(z["pnl_"+target].mean()); tail=float(z["TAIL_"+target].mean())
        feasible=(wr>=.75 and avg>=1.0)
        cand.append(((1 if feasible else 0,wilson(k,n),avg,-tail,n,wr),dict(q=q,th=th,n=n,k=k,wr=wr,avg=avg,tail=tail,feasible=feasible)))
    return max(cand,key=lambda x:x[0])[1]

def select_two_gate(score_c,aux_c,target,aux_good="high",min_n=15):
    cand=[]
    for q in [.55,.65,.75,.80,.85,.90,.925]:
      th=pct_thr(score_c,q)
      for qa in [.35,.50,.65,.75]:
        ath=pct_thr(aux_c,qa)
        m=(score_c>=th)&((aux_c>=ath) if aux_good=="high" else (aux_c<=ath))
        z=cal[m]; n=len(z)
        if n<min_n: continue
        k=int(z[target].sum()); wr=k/n; avg=float(z["pnl_"+target].mean()); tail=float(z["TAIL_"+target].mean())
        feasible=(wr>=.75 and avg>=1.0)
        cand.append(((1 if feasible else 0,wilson(k,n),avg,-tail,n,wr),
                     dict(q=q,qa=qa,th=th,ath=ath,n=n,k=k,wr=wr,avg=avg,tail=tail,feasible=feasible)))
    return max(cand,key=lambda x:x[0])[1]

def summarize_model(model_name,target,cal_meta,audit):
    return {
      "model":model_name,"target":target,
      "cal":{"n":cal_meta["n"],"k":cal_meta["k"],"win_rate_pct":round(100*cal_meta["wr"],2),
             "avg_net_pct":round(cal_meta["avg"],3),"tail_loss_pct":round(100*cal_meta["tail"],2),
             "constraints_met":bool(cal_meta["feasible"])},
      "audit":{"n":audit["n"],"k":audit["k"],"win_rate_pct":round(audit["win_rate"],2) if np.isfinite(audit["win_rate"]) else None,
               "wilson90_lb_pct":round(audit["lb"],2) if np.isfinite(audit["lb"]) else None,
               "avg_net_pct":round(audit["avg"],3) if np.isfinite(audit["avg"]) else None,
               "median_net_pct":round(audit["med"],3) if np.isfinite(audit["med"]) else None,
               "tail_loss_pct":round(audit["tail"],2) if np.isfinite(audit["tail"]) else None},
      "yearly":audit["yearly"]
    }

# ---------------- MODEL 1 ----------------
# Precision ensemble only: baseline of the new five-model sequence.
m1_candidates=[]
for target,_,_ in TARGETS:
    S=components[target]
    g=select_threshold(S["pwin_c"],target)
    a=eval_mask(target,S["pwin_t"]>=g["th"])
    m1_candidates.append((g,a,target))
g1,a1,t1=max(m1_candidates,key=lambda x:(1 if x[0]["feasible"] else 0,wilson(x[0]["k"],x[0]["n"]),x[0]["avg"]))
R1=summarize_model("M1_PRECISION_ENSEMBLE",t1,g1,a1)

# ---------------- MODEL 2 ----------------
# Improvement: tail-loss veto.
m2_candidates=[]
for target,_,_ in TARGETS:
    S=components[target]
    score=S["pwin_c"]*(1-S["ptail_c"])
    score_t=S["pwin_t"]*(1-S["ptail_t"])
    g=select_two_gate(score,S["ptail_c"],target,aux_good="low")
    a=eval_mask(target,(score_t>=g["th"])&(S["ptail_t"]<=g["ath"]))
    m2_candidates.append((g,a,target))
g2,a2,t2=max(m2_candidates,key=lambda x:(1 if x[0]["feasible"] else 0,wilson(x[0]["k"],x[0]["n"]),x[0]["avg"]))
R2=summarize_model("M2_TAIL_VETO",t2,g2,a2)

# ---------------- MODEL 3 ----------------
# Improvement: regime-specific mixture of experts + tail penalty.
m3_candidates=[]
for target,_,_ in TARGETS:
    S=components[target]
    score=S["preg_c"]*(1-S["ptail_c"])*np.sqrt(np.clip(S["ph_c"],1e-9,1))
    score_t=S["preg_t"]*(1-S["ptail_t"])*np.sqrt(np.clip(S["ph_t"],1e-9,1))
    g=select_threshold(score,target)
    a=eval_mask(target,score_t>=g["th"])
    m3_candidates.append((g,a,target))
g3,a3,t3=max(m3_candidates,key=lambda x:(1 if x[0]["feasible"] else 0,wilson(x[0]["k"],x[0]["n"]),x[0]["avg"]))
R3=summarize_model("M3_REGIME_MIXTURE",t3,g3,a3)

# ---------------- MODEL 4 ----------------
# Improvement: CVaR/quantile utility. Penalize bad lower-tail forecast and require mean-return strength.
m4_candidates=[]
for target,_,_ in TARGETS:
    S=components[target]
    # normalized monotone utility score
    scale_mu=1/(1+np.exp(-S["mu_c"]/2)); scale_q=1/(1+np.exp(-S["q20_c"]/3))
    scale_mu_t=1/(1+np.exp(-S["mu_t"]/2)); scale_q_t=1/(1+np.exp(-S["q20_t"]/3))
    score=S["pwin_c"]*(1-S["ptail_c"])*S["ph_c"]*scale_mu*scale_q
    score_t=S["pwin_t"]*(1-S["ptail_t"])*S["ph_t"]*scale_mu_t*scale_q_t
    g=select_two_gate(score,S["mu_c"],target,aux_good="high")
    a=eval_mask(target,(score_t>=g["th"])&(S["mu_t"]>=g["ath"]))
    m4_candidates.append((g,a,target))
g4,a4,t4=max(m4_candidates,key=lambda x:(1 if x[0]["feasible"] else 0,wilson(x[0]["k"],x[0]["n"]),x[0]["avg"]))
R4=summarize_model("M4_CVAR_UTILITY",t4,g4,a4)

# ---------------- MODEL 5 ----------------
# Final improvement: consensus of all independent signals + disagreement veto.
m5_candidates=[]
for target,_,_ in TARGETS:
    S=components[target]
    raw=[
      S["pwin_c"],
      S["pwin_c"]*(1-S["ptail_c"]),
      S["preg_c"]*(1-S["ptail_c"])*np.sqrt(np.clip(S["ph_c"],1e-9,1)),
      S["pwin_c"]*(1-S["ptail_c"])*S["ph_c"]*(1/(1+np.exp(-S["mu_c"]/2)))*(1/(1+np.exp(-S["q20_c"]/3)))
    ]
    raw_t=[
      S["pwin_t"],
      S["pwin_t"]*(1-S["ptail_t"]),
      S["preg_t"]*(1-S["ptail_t"])*np.sqrt(np.clip(S["ph_t"],1e-9,1)),
      S["pwin_t"]*(1-S["ptail_t"])*S["ph_t"]*(1/(1+np.exp(-S["mu_t"]/2)))*(1/(1+np.exp(-S["q20_t"]/3)))
    ]
    # convert each to percentile on 2019 for scale invariance
    ranks=[]; ranks_t=[]
    for a,b in zip(raw,raw_t):
        ss=np.sort(a)
        ranks.append(np.array([(np.searchsorted(ss,v,side="right")+.5)/(len(ss)+1) for v in a]))
        ranks_t.append(np.array([(np.searchsorted(ss,v,side="right")+.5)/(len(ss)+1) for v in b]))
    R=np.vstack(ranks); RT=np.vstack(ranks_t)
    consensus=np.exp(np.mean(np.log(np.clip(R,1e-6,1)),axis=0))
    consensus_t=np.exp(np.mean(np.log(np.clip(RT,1e-6,1)),axis=0))
    disagree=np.std(R,axis=0); disagree_t=np.std(RT,axis=0)
    # low disagreement is good
    g=select_two_gate(consensus,-disagree,target,aux_good="high")
    a=eval_mask(target,(consensus_t>=g["th"])&((-disagree_t)>=g["ath"]))
    m5_candidates.append((g,a,target))
g5,a5,t5=max(m5_candidates,key=lambda x:(1 if x[0]["feasible"] else 0,wilson(x[0]["k"],x[0]["n"]),x[0]["avg"]))
R5=summarize_model("M5_ROBUST_CONSENSUS",t5,g5,a5)

results=[R1,R2,R3,R4,R5]

def promotion(r):
    a=r["audit"]; yrs=[y for y in r["yearly"] if y["n"]>0]
    return (a["n"]>=100 and a["win_rate_pct"]>=75 and a["avg_net_pct"]>=1.0 and
            a["tail_loss_pct"]<=10 and len(yrs)>=5 and
            min(y["rate"] for y in yrs)>=65 and min(y["avg_net"] for y in yrs)>=0)

for r in results: r["promotion_pass"]=promotion(r)
passing=[r for r in results if r["promotion_pass"]]
champ=max(passing,key=lambda r:(r["audit"]["win_rate_pct"],r["audit"]["avg_net_pct"],r["audit"]["n"])) if passing else None

result={
 "study":"KQ Five-Model Iterative Research v1",
 "hard_goal":{"win_rate_pct":75,"avg_net_pct":1.0,"min_signals":100,"tail_loss_pct_max":10},
 "split":{"train":"2015-2018","calibration":"2019","historical_audit":"2020-2026"},
 "note":"2020-2026 is historical audit, not pristine OOS, because earlier research has already examined this era. No 2020+ labels were used for fitting or threshold selection in this run.",
 "models":results,
 "champion":champ,
 "pass_any":bool(champ)
}
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
open("five_model_iterative_v1_result.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
