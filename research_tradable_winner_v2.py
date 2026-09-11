import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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

# ---------- market state ----------
g=d.groupby("Date",sort=True)
macro=g.agg(
    breadth=("ChangesRatio",lambda s:float((s>0).mean())),
    up5=("ChangesRatio",lambda s:float((s>=5).mean())),
    dn5=("ChangesRatio",lambda s:float((s<=-5).mean())),
    medret=("ChangesRatio","median"),
    eqret=("ChangesRatio","mean"),
    mkt_amount=("Amount","sum"),
    lucount=("is_lu","sum"),
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
M=macro.to_dict("index")

d["mcap_pct"]=d.groupby("Date")["Marcap"].rank(pct=True,ascending=False)
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
        if i-last<=4: continue
        last=i
        if i<120 or i+25>=len(gg): continue
        date=pd.Timestamp(dt[i]); mm=M.get(date)
        if not mm: continue

        # ---------- PLQ future labels (research heads only) ----------
        z10=np.arange(i+3,i+11)
        z20=np.arange(i+3,i+21)
        A=bool(np.max(hi[z10])>=cl[i]*1.30)
        B=bool(np.max(hi[z20])>=cl[i]*2.00)
        C=bool(np.min(cl[z10])>=cl[i])
        ANY=bool(A or B or C)

        # ---------- executable entry rule ----------
        # First D+1..D+3 OPEN between -5% and +3% of D0 close.
        # If a prior close already violated -5% from D0, structure is considered broken and no later entry is allowed.
        entry_idx=None
        structure_broken=False
        for j in range(i+1,i+4):
            if j>i+1 and cl[j-1] <= cl[i]*0.95:
                structure_broken=True
            if structure_broken:
                break
            g0=(op[j]/cl[i]-1)*100
            if -5.0 <= g0 <= 3.0:
                entry_idx=j
                break
        tradable=entry_idx is not None

        entry_day=0; entry_gap=np.nan; entry_gap_prev=np.nan
        pre_max=np.nan; pre_min=np.nan; pre_last=np.nan; pre_min_close=np.nan
        TW10=TW15=TW20=TW30=0
        ret10_net=np.nan; exit_reason="NO_ENTRY"; max_run=np.nan; max_dd_close=np.nan

        if tradable:
            e=entry_idx; entry=op[e]; entry_day=e-i
            entry_gap=(entry/cl[i]-1)*100
            entry_gap_prev=(entry/cl[e-1]-1)*100
            if e==i+1:
                pre_max=pre_min=pre_last=pre_min_close=0.0
            else:
                p=np.arange(i+1,e)
                pre_max=(np.max(hi[p])/cl[i]-1)*100
                pre_min=(np.min(lo[p])/cl[i]-1)*100
                pre_last=(cl[e-1]/cl[i]-1)*100
                pre_min_close=(np.min(cl[p])/cl[i]-1)*100

            # helper: target reached before first close-stop; close-stop is evaluated at the close,
            # so an intraday target on that same day is executable before the close stop.
            def target_success(target_pct,horizon):
                end=min(e+horizon-1,len(gg)-1)
                for j in range(e,end+1):
                    if hi[j] >= entry*(1+target_pct/100.0):
                        return 1,j
                    if cl[j] <= entry*0.95:
                        return 0,j
                return 0,end

            TW10,t10j=target_success(10,10)
            TW15,t15j=target_success(15,15)
            TW20,t20j=target_success(20,20)
            TW30,t30j=target_success(30,20)

            hz=np.arange(e,min(e+10,len(gg)))
            max_run=(np.max(hi[hz])/entry-1)*100
            max_dd_close=(np.min(cl[hz])/entry-1)*100

            # executable P&L for primary +10/-5/horizon-10 rule, estimated round-trip cost 0.30%.
            end=min(e+9,len(gg)-1)
            hit=False; stopped=False
            for j in range(e,end+1):
                if hi[j]>=entry*1.10:
                    gross=10.0; hit=True; exit_reason="TP10"; break
                if cl[j]<=entry*0.95:
                    gross=-5.0; stopped=True; exit_reason="STOP_CLOSE5"; break
            if not hit and not stopped:
                gross=(cl[end]/entry-1)*100
                exit_reason="TIME10"
            ret10_net=gross-0.30

        # ---------- D0 features ----------
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
          "mcap_log":math.log1p(mc[i]),"shares_log":math.log1p(st[i]),"mcap_pct":mp[i],
          "turnover_shares":vo[i]/st[i],"turnover_shares_ratio20":(vo[i]/st[i])/turn20 if turn20>0 else np.nan,
          "amount_to_mcap":am[i]/mc[i],
          "pre_ret1":pre1,"pre_ret5":pre5,"pre_ret20":pre20,
          "ret5":r5,"ret10":r10,"ret20":r20,"ret60":r60,"ret120":r120,
          "accel5":r5-r20/4.0,"accel10":r10-r60/6.0,
          "rv10":rv10,"rv20":rv20,
          "high20_dist":cl[i]/ph20-1,"high60_dist":cl[i]/ph60-1,"high120_dist":cl[i]/ph120-1,
          "low20_dist":cl[i]/pl20-1,"low60_dist":cl[i]/pl60-1,
          "ma20_dist":cl[i]/ma20-1,"ma60_dist":cl[i]/ma60-1,
          "prior_lu20":float(np.sum(lu[i-20:i])),"prior_lu60":float(np.sum(lu[i-60:i])),"prior_lu120":float(np.sum(lu[i-120:i])),
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
          "M_breadth":mm["breadth"],"M_breadth5":mm["breadth5"],"M_breadth20":mm["breadth20"],
          "M_breadth_accel":mm["breadth_accel"],"M_up5":mm["up5"],"M_up5_5":mm["up5_5"],
          "M_momentum_diffusion":mm["momentum_diffusion"],"M_risk_dispersion":mm["risk_dispersion"],
          "M_eqret5":mm["eqret5"],"M_eqret20":mm["eqret20"],"M_amount_shock":mm["amount_shock"],
          "M_lucount":mm["lucount"],"M_lucount5":mm["lucount5"],"M_lu_accel":mm["lu_accel"],
        }
        if any(not np.isfinite(v) for v in vals.values()): continue

        rec={"Code":code,"Name":nm[i],"Date":date,"year":date.year,
             "A":int(A),"B":int(B),"C":int(C),"ANY":int(ANY),
             "tradable":int(tradable),"entry_day":int(entry_day),
             "entry_gap":float(entry_gap) if tradable else np.nan,
             "entry_gap_prev":float(entry_gap_prev) if tradable else np.nan,
             "preentry_max":float(pre_max) if tradable else np.nan,
             "preentry_min":float(pre_min) if tradable else np.nan,
             "preentry_last":float(pre_last) if tradable else np.nan,
             "preentry_min_close":float(pre_min_close) if tradable else np.nan,
             "TW10":int(TW10),"TW15":int(TW15),"TW20":int(TW20),"TW30":int(TW30),
             "ret10_net":float(ret10_net) if tradable else np.nan,
             "exit_reason":exit_reason,
             "max_run":float(max_run) if tradable else np.nan,
             "max_dd_close":float(max_dd_close) if tradable else np.nan}
        rec.update({k:float(v) for k,v in vals.items()})
        events.append(rec)

ev=pd.DataFrame(events).sort_values("Date").reset_index(drop=True)

D0=[c for c in ev.columns if c not in [
    "Code","Name","Date","year","A","B","C","ANY","tradable","entry_day","entry_gap","entry_gap_prev",
    "preentry_max","preentry_min","preentry_last","preentry_min_close",
    "TW10","TW15","TW20","TW30","ret10_net","exit_reason","max_run","max_dd_close"
]]
ENTRY=["entry_day","entry_gap","entry_gap_prev","preentry_max","preentry_min","preentry_last","preentry_min_close"]

# Only actually executable entries are used for the profit model.
train=ev[(ev.year>=2015)&(ev.year<=2018)&(ev.tradable==1)].copy()
cal=ev[(ev.year==2019)&(ev.tradable==1)].copy()
test=ev[(ev.year>=2020)&(ev.year<=2026)&(ev.tradable==1)].copy()

print("EVENTS",len(ev),"TRADABLE",len(train),len(cal),len(test),flush=True)
print("TEST_BASE", {k:round(100*test[k].mean(),2) for k in ["TW10","TW15","TW20","TW30","ANY"]},flush=True)

def hgb():
    return HistGradientBoostingClassifier(max_depth=3,learning_rate=.035,max_iter=240,l2_regularization=6,min_samples_leaf=15,random_state=SEED)

def logit():
    return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.18,max_iter=7000,class_weight="balanced",random_state=SEED))])

def wilson(k,n,z=1.6448536269514722):
    if n<=0:return 0.0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def choose(cp,target,min_n=20):
    opts=[]
    for q in [0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.925,0.95,0.96,0.97]:
        th=float(np.quantile(cp,q))
        s=cal[cp>=th]; n=len(s); k=int(s[target].sum())
        if n>=min_n:
            opts.append((wilson(k,n),k/n,n,q,th,k))
    return max(opts,key=lambda x:(x[0],x[1],x[2]))

def report(name,cp,tp,target="TW10"):
    lb,p,n,q,th,k=choose(cp,target)
    s=test[tp>=th].copy(); N=len(s); K=int(s[target].sum())
    yr=[]
    for y in range(2020,2027):
        z=s[s.year==y]; nn=len(z); kk=int(z[target].sum())
        yr.append({"year":y,"n":nn,"k":kk,"rate":round(100*kk/nn,2) if nn else None,
                   "avg_net":round(float(z.ret10_net.mean()),2) if nn else None})
    a,b=1+K,1+N-K
    return {
      "model":name,"target":target,
      "cal":{"q":q,"n":n,"k":k,"rate":round(100*p,2),"wilson90_lb":round(100*lb,2)},
      "test":{"n":N,"k":K,"rate":round(100*K/N,2) if N else None,
              "wilson90_lb":round(100*wilson(K,N),2) if N else None,
              "beta90_ci":[round(100*beta_dist.ppf(.05,a,b),2),round(100*beta_dist.ppf(.95,a,b),2)] if N else None,
              "avg_net_pct":round(float(s.ret10_net.mean()),3) if N else None,
              "median_net_pct":round(float(s.ret10_net.median()),3) if N else None,
              "positive_net_pct":round(100*float((s.ret10_net>0).mean()),2) if N else None,
              "TW15_pct":round(100*s.TW15.mean(),2) if N else None,
              "TW20_pct":round(100*s.TW20.mean(),2) if N else None,
              "TW30_pct":round(100*s.TW30.mean(),2) if N else None,
              "ANY_PLQ_pct":round(100*s.ANY.mean(),2) if N else None,
              "avg_max_run_pct":round(float(s.max_run.mean()),2) if N else None,
              "avg_max_dd_close_pct":round(float(s.max_dd_close.mean()),2) if N else None},
      "yearly":yr,
      "threshold":th
    }

results=[]

# 1. D0-only: can be known on limit-up close.
m0=hgb(); m0.fit(train[D0],train.TW10)
cp0=m0.predict_proba(cal[D0])[:,1]; tp0=m0.predict_proba(test[D0])[:,1]
r0=report("TW_D0_HGB",cp0,tp0); results.append(r0)

# 2. Entry-time direct model: adds only information known at the chosen D+1..D+3 open.
X=D0+ENTRY
m1=hgb(); m1.fit(train[X],train.TW10)
cp1=m1.predict_proba(cal[X])[:,1]; tp1=m1.predict_proba(test[X])[:,1]
r1=report("TW_ENTRY_HGB",cp1,tp1); results.append(r1)

# 3. Regularized entry-time logit robustness.
ml=logit(); ml.fit(train[X],train.TW10)
cpl=ml.predict_proba(cal[X])[:,1]; tpl=ml.predict_proba(test[X])[:,1]
rl=report("TW_ENTRY_LOGIT",cpl,tpl); results.append(rl)

# 4. PLQ latent heads without future leakage in final model training:
#    OOF head predictions for 2016-2018; final entry model learns on those OOF rows.
oof_parts=[]
for vy in [2016,2017,2018]:
    tr=ev[(ev.year>=2015)&(ev.year<vy)]
    va=ev[(ev.year==vy)&(ev.tradable==1)].copy()
    if len(tr)<100 or len(va)==0: continue
    pp=[]
    for tgt in ["A","B","C"]:
        mh=hgb(); mh.fit(tr[D0],tr[tgt])
        pp.append(mh.predict_proba(va[D0])[:,1])
    va["pA"],va["pB"],va["pC"]=pp[0],pp[1],pp[2]
    oof_parts.append(va)
oof=pd.concat(oof_parts,ignore_index=True)

# Heads refit only on <=2018 and then frozen for 2019/2020+.
heads={}
for tgt in ["A","B","C"]:
    mh=hgb(); mh.fit(ev[(ev.year>=2015)&(ev.year<=2018)][D0],ev[(ev.year>=2015)&(ev.year<=2018)][tgt]); heads[tgt]=mh
for z in [cal,test]:
    z["pA"]=heads["A"].predict_proba(z[D0])[:,1]
    z["pB"]=heads["B"].predict_proba(z[D0])[:,1]
    z["pC"]=heads["C"].predict_proba(z[D0])[:,1]
    z["pANY_ind"]=1-(1-z.pA)*(1-z.pB)*(1-z.pC)
    z["pPLQ_geo"]=np.cbrt(np.clip(z.pA*z.pB*z.pC,1e-12,1))
oof["pANY_ind"]=1-(1-oof.pA)*(1-oof.pB)*(1-oof.pC)
oof["pPLQ_geo"]=np.cbrt(np.clip(oof.pA*oof.pB*oof.pC,1e-12,1))

LAT=["pA","pB","pC","pANY_ind","pPLQ_geo"]
metaX=ENTRY+LAT
mm=hgb(); mm.fit(oof[metaX],oof.TW10)
cpm=mm.predict_proba(cal[metaX])[:,1]; tpm=mm.predict_proba(test[metaX])[:,1]
rm=report("TW_PLQ_LATENT_ENTRY",cpm,tpm); results.append(rm)

# 5. Conservative consensus of direct entry model and latent model via percentile ranks.
def pct_rank(cal_score,x):
    s=np.sort(cal_score)
    return (np.searchsorted(s,x,side="right")+.5)/(len(s)+1.0)
c1=np.array([pct_rank(cp1,v) for v in cp1]); t1=np.array([pct_rank(cp1,v) for v in tp1])
cm=np.array([pct_rank(cpm,v) for v in cpm]); tm=np.array([pct_rank(cpm,v) for v in tpm])
cc=np.sqrt(c1*cm); tt=np.sqrt(t1*tm)
rc=report("TW_GEOMETRIC_CONSENSUS",cc,tt); results.append(rc)

# Fixed top-10 / top-5 robustness for consensus, independent of outcome threshold optimization.
robust={}
for q in [.90,.95]:
    th=float(np.quantile(cc,q))
    s=test[tt>=th]; n=len(s); k=int(s.TW10.sum())
    robust[str(q)]={"n":n,"k":k,"rate":round(100*k/n,2) if n else None,
                    "avg_net_pct":round(float(s.ret10_net.mean()),3) if n else None,
                    "positive_net_pct":round(100*float((s.ret10_net>0).mean()),2) if n else None}

best=max([r for r in results if r["test"]["n"]>=100],key=lambda r:(r["test"]["rate"],r["test"]["wilson90_lb"]))

result={
 "study":"PLQ Tradable Winner v2",
 "primary_trade_rule":{
   "entry":"first D+1..D+3 open in [-5%, +3%] vs D0 limit-up close; prior -5% close break cancels later entry",
   "target":"+10% intraday high","stop":"-5% close","horizon":"10 trading days from entry",
   "cost":"0.30% round-trip estimate","RR":"2.0 nominal"
 },
 "secondary_labels":{"TW15":"+15% before same stop within 15d","TW20":"+20% before same stop within 20d","TW30":"+30% before same stop within 20d"},
 "split":{"train":"2015-2018","calibration":"2019","untouched_test":"2020-2026"},
 "events":{"all":len(ev),"train_tradable":len(train),"cal_tradable":len(cal),"test_tradable":len(test),
           "test_entry_opportunity_pct":round(100*float(ev[(ev.year>=2020)&(ev.year<=2026)].tradable.mean()),2)},
 "test_base_rates_pct":{k:round(100*float(test[k].mean()),2) for k in ["TW10","TW15","TW20","TW30","ANY"]},
 "models":results,
 "consensus_fixed_selectivity":robust,
 "best_model":best,
 "promotion_rule":{"minimum_oos_signals":100,"TW10_win_rate_target_pct":75,"minimum_avg_net_pct":0},
 "pass":bool(best["test"]["n"]>=100 and best["test"]["rate"]>=75 and best["test"]["avg_net_pct"]>0)
}
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
open("tradable_winner_v2_result.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
