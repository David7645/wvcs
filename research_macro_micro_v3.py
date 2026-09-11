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
        cont=ptype in (1,2); pos=bool(tradable and cont and tp)
        rec={"Code":code,"Name":nm[i],"Date":date,"year":date.year,"d1_gap":float(d1_gap),"path_type":ptype,
             "continuation":int(cont),"positive":int(pos),"tradable":int(tradable),"mfe":mfe,"mae":mae,"d5ret":d5ret}
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

def model():
    return HistGradientBoostingClassifier(max_depth=3,learning_rate=.035,max_iter=180,l2_regularization=4,min_samples_leaf=12,random_state=SEED)

def wilson(k,n,z=1.6448536269514722):
    if n<=0:return 0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

train=ev[(ev.year>=2015)&(ev.year<=2018)&(ev.tradable==1)]
cal=ev[(ev.year==2019)&(ev.tradable==1)]
test=ev[(ev.year>=2020)&(ev.year<=2026)&(ev.tradable==1)]
print("EVENTS",len(ev),"SPLITS",len(train),len(cal),len(test),flush=True)

sets={
 "MACRO_ONLY":macro_cols,
 "MESO_MICRO_PROXY":meso_cols+micro_cols,
 "MACRO_PLUS_MICRO":combined_cols,
}

def evaluate(name,cols,target):
    m=model(); m.fit(train[cols],train[target])
    cp=m.predict_proba(cal[cols])[:,1]
    tp=m.predict_proba(test[cols])[:,1]
    opts=[]
    # Pre-specified selectivity grid only; all choice done on 2019.
    for q in [0.70,0.75,0.80,0.85,0.90,0.925,0.95]:
        th=float(np.quantile(cp,q)); s=cal[cp>=th]; n=len(s); k=int(s[target].sum())
        if n>=12: opts.append((wilson(k,n),k/n,n,q,th))
    best=max(opts,key=lambda z:(z[0],z[1],z[2]))
    lb,prec,n,q,th=best
    s=test[tp>=th].copy(); N=len(s); K=int(s[target].sum())
    yearly=[]
    for y in range(2020,2027):
        sy=s[s.year==y]; nn=len(sy); kk=int(sy[target].sum())
        yearly.append({"year":y,"n":nn,"k":kk,"rate":round(100*kk/nn,2) if nn else None})
    out={"name":name,"target":target,"features":len(cols),
         "cal":{"q":q,"threshold":round(th,6),"n":n,"k":k,"rate":round(prec*100,2),"lb":round(lb*100,2)},
         "test":{"n":N,"k":K,"rate":round(100*K/N,2) if N else None,"lb":round(wilson(K,N)*100,2) if N else None,
                 "avg_mfe":round(float(s.mfe.mean()),2) if N else None,"avg_mae":round(float(s.mae.mean()),2) if N else None,
                 "avg_d5":round(float(s.d5ret.mean()),2) if N else None},
         "yearly":yearly}
    return out,m,cp,tp,th

results=[]
art={}
for name,cols in sets.items():
    for target in ["continuation","positive"]:
        o,*rest=evaluate(name,cols,target); results.append(o); art[(name,target)]=(o,*rest)

# Hierarchical macro gate -> combined selector. Both thresholds fixed on 2019.
mm=model(); mm.fit(train[macro_cols],train.continuation)
cm=model(); cm.fit(train[combined_cols],train.positive)
cpm=mm.predict_proba(cal[macro_cols])[:,1]; cpc=cm.predict_proba(cal[combined_cols])[:,1]
tpm=mm.predict_proba(test[macro_cols])[:,1]; tpc=cm.predict_proba(test[combined_cols])[:,1]
opts=[]
for qm in [0.50,0.60,0.70,0.80]:
  tm=float(np.quantile(cpm,qm))
  for qx in [0.70,0.80,0.85,0.90,0.925,0.95]:
    tx=float(np.quantile(cpc,qx))
    mask=(cpm>=tm)&(cpc>=tx)
    s=cal[mask]; n=len(s); k=int(s.positive.sum())
    if n>=12: opts.append((wilson(k,n),k/n,n,qm,qx,tm,tx))
best=max(opts,key=lambda z:(z[0],z[1],z[2]))
lb,prec,n,qm,qx,tm,tx=best
mask=(tpm>=tm)&(tpc>=tx); s=test[mask].copy(); N=len(s); K=int(s.positive.sum())
hier={"name":"HIERARCHICAL_MACRO_GATE_X_MICRO_SELECTOR","target":"positive","cal":{"qm":qm,"qx":qx,"n":n,"k":k,"rate":round(prec*100,2),"lb":round(lb*100,2)},
      "test":{"n":N,"k":K,"rate":round(100*K/N,2) if N else None,"lb":round(wilson(K,N)*100,2) if N else None,
              "continuation_precision":round(100*s.continuation.mean(),2) if N else None,
              "avg_mfe":round(float(s.mfe.mean()),2) if N else None,"avg_mae":round(float(s.mae.mean()),2) if N else None,
              "avg_d5":round(float(s.d5ret.mean()),2) if N else None},
      "yearly":[]}
for y in range(2020,2027):
    sy=s[s.year==y]; nn=len(sy); kk=int(sy.positive.sum())
    hier["yearly"].append({"year":y,"n":nn,"k":kk,"rate":round(100*kk/nn,2) if nn else None})

# Macro regime prevalence descriptive study, quantile-based, no model claims.
macro_desc=[]
for feat in ["M_breadth5","M_breadth20","M_breadth_accel","M_lucount5","M_lu_accel","M_mkt_amt_ratio20","M_up5_5","M_risk_dispersion"]:
    if feat not in test: continue
    qhi=float(train[feat].quantile(.7)); qlo=float(train[feat].quantile(.3))
    for label,mask in [("LOW",test[feat]<=qlo),("MID",(test[feat]>qlo)&(test[feat]<qhi)),("HIGH",test[feat]>=qhi)]:
        z=test[mask]; n=len(z)
        macro_desc.append({"feature":feat,"bucket":label,"n":n,
                           "continuation_pct":round(100*z.continuation.mean(),2) if n else None,
                           "positive_pct":round(100*z.positive.mean(),2) if n else None})

result={"study":"SANGTTA MACRO x MICRO RESEARCH v3",
        "data_note":"True historical tick/orderbook microstructure is unavailable in marcap; X_* are D0 daily microstructure proxies + D+1 opening gap. Live NH micro fields are validated separately.",
        "events_total":len(ev),"splits":{"train":len(train),"cal":len(cal),"test":len(test)},
        "model_comparison":results,"hierarchical":hier,"macro_regime_descriptives":macro_desc,
        "pass_80":bool(hier["test"]["n"]>=100 and hier["test"]["rate"] is not None and hier["test"]["rate"]>=80)}
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
open("macro_micro_v3_result.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
