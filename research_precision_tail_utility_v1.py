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

# ---------------- Precision + Tail-Risk + Expected-Return research ----------------
# This version is designed around the user's hard constraints:
# win rate >=75%, mean net return >=1%, N>=100 in the 2020-2026 historical audit.

def hgb_c():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=.03, max_iter=260,
        l2_regularization=8, min_samples_leaf=18, random_state=SEED)

def hgb_r(loss="squared_error", quantile=None):
    kw=dict(max_depth=3,learning_rate=.03,max_iter=260,
            l2_regularization=8,min_samples_leaf=18,random_state=SEED,loss=loss)
    if quantile is not None:
        kw["quantile"]=quantile
    return HistGradientBoostingRegressor(**kw)

def elastic():
    return Pipeline([("s",StandardScaler()),
        ("m",LogisticRegression(C=.15,penalty="elasticnet",solver="saga",
             l1_ratio=.20,class_weight="balanced",max_iter=10000,random_state=SEED))])

def lda():
    return Pipeline([("s",StandardScaler()),
        ("m",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])

def wilson(k,n,z=1.6448536269514722):
    if n<=0:return 0.0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

# Tail loss = realized net P/L <= -5%.
# This catches the close-stop overshoot problem that made the 76% win-rate model unprofitable.
for target,_,_ in TARGETS:
    ev["TAIL_"+target]=(ev["pnl_"+target] <= -5.0).astype(int)

# Refresh split frames after tail labels added.
train=ev[(ev.year>=2015)&(ev.year<=2018)].copy()
cal=ev[ev.year==2019].copy()
test=ev[(ev.year>=2020)&(ev.year<=2026)].copy()

def fit_component_scores(target):
    # 1) Win probability ensemble.
    win_models=[hgb_c(),elastic(),lda()]
    cwin=[]; twin=[]
    for m in win_models:
        m.fit(train[features],train[target])
        cwin.append(m.predict_proba(cal[features])[:,1])
        twin.append(m.predict_proba(test[features])[:,1])
    pwin_cal=np.mean(np.vstack(cwin),axis=0)
    pwin_test=np.mean(np.vstack(twin),axis=0)

    # 2) Tail-loss probability.
    tail="TAIL_"+target
    tail_models=[hgb_c(),elastic()]
    ctail=[]; ttail=[]
    for m in tail_models:
        m.fit(train[features],train[tail])
        ctail.append(m.predict_proba(cal[features])[:,1])
        ttail.append(m.predict_proba(test[features])[:,1])
    ptail_cal=np.mean(np.vstack(ctail),axis=0)
    ptail_test=np.mean(np.vstack(ttail),axis=0)

    # 3) Conditional-safety hurdle probability.
    ms=hgb_c(); ms.fit(train[features],train.safe3)
    trsafe=train[train.safe3==1]
    mt=hgb_c(); mt.fit(trsafe[features],trsafe[target])
    ph_cal=ms.predict_proba(cal[features])[:,1]*mt.predict_proba(cal[features])[:,1]
    ph_test=ms.predict_proba(test[features])[:,1]*mt.predict_proba(test[features])[:,1]

    # 4) Mean P/L regression.
    mr=hgb_r(); mr.fit(train[features],train["pnl_"+target])
    mu_cal=mr.predict(cal[features]); mu_test=mr.predict(test[features])

    # 5) Lower-tail quantile P/L regression (20th percentile).
    qr=hgb_r(loss="quantile",quantile=.20); qr.fit(train[features],train["pnl_"+target])
    q20_cal=qr.predict(cal[features]); q20_test=qr.predict(test[features])

    # 6) Economic-regime / liquidity variables are not learned from test outcomes.
    return {
      "pwin_cal":pwin_cal,"pwin_test":pwin_test,
      "ptail_cal":ptail_cal,"ptail_test":ptail_test,
      "ph_cal":ph_cal,"ph_test":ph_test,
      "mu_cal":mu_cal,"mu_test":mu_test,
      "q20_cal":q20_cal,"q20_test":q20_test
    }

def pct_threshold(x,q):
    return float(np.quantile(x,q))

def select_gate(target,S):
    # Search only 2019. No 2020+ labels are touched here.
    # Gate families are deliberately coarse to reduce calibration overfit.
    candidates=[]
    for qw in [.55,.65,.75,.80,.85,.90]:
      tw=pct_threshold(S["pwin_cal"],qw)
      for qh in [.50,.65,.75,.85]:
        th=pct_threshold(S["ph_cal"],qh)
        for qt in [.40,.55,.70]:
          # low tail probability is good: use lower quantile cutoff
          tt=pct_threshold(S["ptail_cal"],qt)
          for qm in [.40,.55,.70]:
            tm=pct_threshold(S["mu_cal"],qm)
            for qq in [.30,.45,.60]:
              tq=pct_threshold(S["q20_cal"],qq)
              for qrisk in [.30,.45,.60]:
                tr=pct_threshold(cal.M_risk_on.to_numpy(),qrisk)
                mask=(S["pwin_cal"]>=tw)&(S["ph_cal"]>=th)&(S["ptail_cal"]<=tt)&(S["mu_cal"]>=tm)&(S["q20_cal"]>=tq)&(cal.M_risk_on.to_numpy()>=tr)
                z=cal[mask]; n=len(z)
                if n<15: continue
                k=int(z[target].sum()); wr=k/n; avg=float(z["pnl_"+target].mean())
                tail=float(z["TAIL_"+target].mean())
                # Hard development constraints first. If none satisfy them, keep near-misses separately.
                feasible=(wr>=.75 and avg>=1.0)
                obj=(1 if feasible else 0, wilson(k,n), avg, -tail, n, wr)
                candidates.append((obj,{"qw":qw,"qh":qh,"qt":qt,"qm":qm,"qq":qq,"qrisk":qrisk,
                    "tw":tw,"th":th,"tt":tt,"tm":tm,"tq":tq,"tr":tr,
                    "cal_n":n,"cal_k":k,"cal_wr":wr,"cal_avg":avg,"cal_tail":tail,
                    "feasible":feasible}))
    if not candidates: raise RuntimeError("no calibration gate candidates")
    candidates.sort(key=lambda x:x[0],reverse=True)
    return candidates[0][1], candidates[:10]

def apply_gate(df,S,gate,side):
    return (
      (S["pwin_"+side]>=gate["tw"]) &
      (S["ph_"+side]>=gate["th"]) &
      (S["ptail_"+side]<=gate["tt"]) &
      (S["mu_"+side]>=gate["tm"]) &
      (S["q20_"+side]>=gate["tq"]) &
      (df.M_risk_on.to_numpy()>=gate["tr"])
    )

def summarize(target,S,gate):
    m=apply_gate(test,S,gate,"test"); z=test[m].copy()
    N=len(z); K=int(z[target].sum())
    yrs=[]
    for y in range(2020,2027):
        a=z[z.year==y]; n=len(a); k=int(a[target].sum())
        yrs.append({
          "year":y,"n":n,"k":k,
          "rate":round(100*k/n,2) if n else None,
          "avg_net":round(float(a["pnl_"+target].mean()),3) if n else None,
          "tail_loss_pct":round(100*float(a["TAIL_"+target].mean()),2) if n else None
        })
    return {
      "target":target,
      "calibration":{
        "n":gate["cal_n"],"k":gate["cal_k"],
        "win_rate_pct":round(100*gate["cal_wr"],2),
        "avg_net_pct":round(gate["cal_avg"],3),
        "tail_loss_pct":round(100*gate["cal_tail"],2),
        "development_constraints_met":bool(gate["feasible"])
      },
      "audit":{
        "n":N,"k":K,
        "win_rate_pct":round(100*K/N,2) if N else None,
        "wilson90_lb_pct":round(100*wilson(K,N),2) if N else None,
        "avg_net_pct":round(float(z["pnl_"+target].mean()),3) if N else None,
        "median_net_pct":round(float(z["pnl_"+target].median()),3) if N else None,
        "tail_loss_pct":round(100*float(z["TAIL_"+target].mean()),2) if N else None,
        "positive_net_pct":round(100*float((z["pnl_"+target]>0).mean()),2) if N else None,
        "avg_pred_win":round(float(S["pwin_test"][m].mean()),3) if N else None,
        "avg_pred_tail":round(float(S["ptail_test"][m].mean()),3) if N else None,
        "avg_pred_mu":round(float(S["mu_test"][m].mean()),3) if N else None,
        "avg_pred_q20":round(float(S["q20_test"][m].mean()),3) if N else None
      },
      "yearly":yrs,
      "gate":{k:(round(float(v),6) if isinstance(v,(float,np.floating)) else v)
              for k,v in gate.items() if k not in ["cal_n","cal_k","cal_wr","cal_avg","cal_tail","feasible"]}
    }

target_results=[]
dev_tops={}
scores={}
for target,_,_ in TARGETS:
    S=fit_component_scores(target)
    gate,top10=select_gate(target,S)
    target_results.append(summarize(target,S,gate))
    dev_tops[target]=[
      {"rank":i+1,"feasible":bool(g["feasible"]),"cal_n":g["cal_n"],
       "cal_win_rate_pct":round(100*g["cal_wr"],2),
       "cal_avg_net_pct":round(g["cal_avg"],3),
       "cal_tail_loss_pct":round(100*g["cal_tail"],2)}
      for i,(_,g) in enumerate(top10[:5])
    ]
    scores[target]=S

# Choose the target using ONLY 2019 development performance:
# feasible first, then Wilson lower bound proxy via observed win rate/support, then avg return.
def choose_key(r):
    c=r["calibration"]
    return (1 if c["development_constraints_met"] else 0,
            c["win_rate_pct"],c["avg_net_pct"],c["n"])
chosen=max(target_results,key=choose_key)
chosen_target=chosen["target"]

# Promotion is judged only on historical audit, with user's hard constraints.
def audit_pass(r):
    a=r["audit"]
    if a["n"] is None: return False
    yearly=[y for y in r["yearly"] if y["n"]>0]
    return (
      a["n"]>=100 and
      a["win_rate_pct"]>=75 and
      a["avg_net_pct"]>=1.0 and
      a["tail_loss_pct"]<=10 and
      len(yearly)>=5 and
      min(y["avg_net"] for y in yearly)>=0 and
      min(y["rate"] for y in yearly)>=65
    )

for r in target_results:
    r["promotion_pass"]=audit_pass(r)

passing=[r for r in target_results if r["promotion_pass"]]
champ=max(passing,key=lambda r:(r["audit"]["win_rate_pct"],r["audit"]["avg_net_pct"],r["audit"]["n"])) if passing else None

result={
 "study":"KQ Precision-Tail-Utility v1",
 "goal":{
   "minimum_audit_signals":100,
   "minimum_win_rate_pct":75,
   "minimum_avg_net_pct":1.0,
   "maximum_tail_loss_rate_pct":10,
   "minimum_year_win_rate_pct":65,
   "minimum_year_avg_net_pct":0
 },
 "architecture":[
   "ensemble win probability: HGB + ElasticNet logit + shrinkage LDA",
   "tail-loss classifier for realized net loss <= -5%",
   "safety hurdle: early survival x target probability",
   "mean-PnL regression",
   "20th-percentile PnL quantile regression",
   "economic risk-on/liquidity gate",
   "coarse 2019-only constrained gate search"
 ],
 "entry_rule":"first D+1..D+3 open within -3%..+3% of D0 limit-up close; prior -3% close break cancels later entry",
 "stop":"-3% close-based",
 "cost":"0.30% estimated round-trip",
 "targets":{"T3":"+3% within 5d","T4":"+4% within 7d","T5":"+5% within 10d"},
 "split":{"train":"2015-2018","gate_selection":"2019","historical_audit":"2020-2026"},
 "audit_note":"2020-2026 is not pristine OOS because previous research has already examined it; no 2020+ labels were used in this run's fitting or gate selection.",
 "events":{"train":len(train),"cal":len(cal),"audit":len(test)},
 "target_results":target_results,
 "development_top_candidates":dev_tops,
 "chosen_target_by_2019":chosen_target,
 "champion":champ,
 "pass_any":bool(champ is not None)
}
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
open("precision_tail_utility_v1_result.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
