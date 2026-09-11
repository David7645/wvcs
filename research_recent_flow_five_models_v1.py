import os, json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED=42
FLOW_ROOT="flowdata/archive"
# ---------- load marcap 2026 ----------
cols=["Date","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Stocks","Market"]
d=pd.read_parquet("data/marcap-2026.parquet",columns=cols)
d=d[d.Market.astype(str).str.upper().eq("KOSDAQ")].copy()
d["Date"]=pd.to_datetime(d.Date); d["Code"]=d.Code.astype(str).str.zfill(6)
for c in cols[3:-1]: d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna().sort_values(["Code","Date"]).reset_index(drop=True)
d=d[(d.Open>0)&(d.High>0)&(d.Low>0)&(d.Close>0)&(d.Amount>0)&(d.Stocks>0)]
d["r1"]=d.groupby("Code").Close.pct_change()*100
d["r5"]=d.groupby("Code").Close.pct_change(5)*100
d["r20"]=d.groupby("Code").Close.pct_change(20)*100
d["rv5"]=d.groupby("Code").r1.rolling(5).std().reset_index(level=0,drop=True)
d["vmed20"]=d.groupby("Code").Volume.rolling(20).median().reset_index(level=0,drop=True)
d["vr"]=d.Volume/d.vmed20
d["ap"]=d.groupby("Date").Amount.rank(pct=True,ascending=False)
d["range"]=np.maximum(d.High-d.Low,1e-9)
d["upper"]=(d.High-np.maximum(d.Open,d.Close))/d["range"]
d["clv"]=(d.Close-d.Low)/d["range"]
d["lu"]=(d.ChangesRatio>=29.5)&(d.Close==d.High)

# future trade label at next open
recs=[]
for code,z in d.groupby("Code",sort=False):
    z=z.reset_index()
    O,H,C=z.Open.values,z.High.values,z.Close.values
    for k in range(20,len(z)-6):
        row=z.iloc[k]; dt=pd.Timestamp(row.Date)
        gap=(O[k+1]/C[k]-1)*100
        elig=int(-3<=gap<=3)
        win=0; pnl=np.nan
        if elig:
            e=O[k+1]; gross=None
            for j in range(k+1,min(k+5,len(z)-1)+1):
                if H[j]>=e*1.03:
                    win=1; gross=3.0; break
                if C[j]<=e*.97:
                    gross=(C[j]/e-1)*100; break
            if gross is None: gross=(C[min(k+5,len(z)-1)]/e-1)*100
            pnl=gross-.30
        recs.append({"Date":dt,"Code":code,"elig":elig,"gap":gap,"W3":win,"pW3":pnl,
                     "r1":row.r1,"r5":row.r5,"r20":row.r20,"rv5":row.rv5,"vr":row.vr,
                     "ap":row.ap,"upper":row.upper,"clv":row.clv,"Marcap":row.Marcap,"Stocks":row.Stocks,
                     "Amount":row.Amount,"lu":int(row.lu)})
bar=pd.DataFrame(recs).dropna(subset=["r1","r5","r20","rv5","vr"])

# ---------- load archived investor-flow snapshots ----------
snap=[]
sec=[]
for folder in sorted(os.listdir(FLOW_ROOT)):
    p=os.path.join(FLOW_ROOT,folder)
    if not os.path.isdir(p): continue
    jf=os.path.join(p,"foreign-flows-latest.json")
    cf=os.path.join(p,"foreign-flows-aggregates.csv")
    sf=os.path.join(p,"korea-sectors-latest.csv")
    if not (os.path.exists(jf) and os.path.exists(cf)): continue
    try:
        meta=json.load(open(jf,encoding="utf-8"))["meta"]
        asof=pd.Timestamp(meta["as_of_trading_day"])
        x=pd.read_csv(cf,comment="#",dtype={"ticker":str})
        x=x[x.market.eq("KOSDAQ")].copy()
        x["Code"]=x.ticker.astype(str).str.zfill(6); x["Date"]=asof
        snap.append(x)
        if os.path.exists(sf):
            s=pd.read_csv(sf,comment="#")
            s=s[s.market.eq("KOSDAQ") & ~s.sector_en.str.contains("KOSDAQ",case=False,na=False)].copy()
            if len(s):
                sec.append({"Date":asof,
                    "sec_pos1":float((s.return_1d_pct>0).mean()),
                    "sec_mean1":float(s.return_1d_pct.mean()),
                    "sec_disp1":float(s.return_1d_pct.std()),
                    "sec_best1":float(s.return_1d_pct.max()),
                    "sec_pos1m":float((s.return_1m_pct>0).mean()),
                    "sec_excess1m":float(s.excess_1m_vs_market_pp.mean()),
                    "sec_best_excess1m":float(s.excess_1m_vs_market_pp.max())})
    except Exception as e:
        print("skip",folder,repr(e))
flows=pd.concat(snap,ignore_index=True).drop_duplicates(["Date","Code"],keep="last")
sectors=pd.DataFrame(sec).drop_duplicates("Date",keep="last")
print("SNAPSHOTS",flows.Date.nunique(),flows.Date.min(),flows.Date.max(),flush=True)

# join and restrict to fully observable horizon through 2026-09-04
x=bar.merge(flows,on=["Date","Code"],how="inner").merge(sectors,on="Date",how="left")
x=x[(x.Date<=pd.Timestamp("2026-09-04")) & (x.lu==0) & (x.ap<=.30) & (x.Amount>=1e9) & (x.upper<=.60)].copy()
for c in ["f_1d_sh","f_5d_sh","f_20d_sh","i_1d_sh","i_5d_sh","i_20d_sh","f_5d_val","f_20d_val"]:
    x[c]=pd.to_numeric(x[c],errors="coerce").fillna(0)
# normalize flows
x["f1"]=x.f_1d_sh/x.Stocks; x["f5"]=x.f_5d_sh/x.Stocks; x["f20"]=x.f_20d_sh/x.Stocks
x["i1"]=x.i_1d_sh/x.Stocks; x["i5"]=x.i_5d_sh/x.Stocks; x["i20"]=x.i_20d_sh/x.Stocks
x["fv5mc"]=x.f_5d_val/x.Marcap; x["fv20mc"]=x.f_20d_val/x.Marcap
x["f_accel1"]=x.f1-x.f5/5; x["f_accel5"]=x.f5/5-x.f20/20
x["i_accel1"]=x.i1-x.i5/5; x["i_accel5"]=x.i5/5-x.i20/20
x["smart5"]=x.f5+x.i5; x["smart20"]=x.f20+x.i20
x["cons1"]=((x.f1>0)&(x.i1>0)).astype(int); x["cons5"]=((x.f5>0)&(x.i5>0)).astype(int)
x["absorb1"]=x.f1*np.maximum(-x.r1,0); x["absorb5"]=x.f5*np.maximum(-x.r5,0)
x=x.replace([np.inf,-np.inf],np.nan).dropna()
print("JOINED",len(x),"dates",x.Date.nunique(),flush=True)

PRICE=["r1","r5","r20","rv5","vr","upper","clv","ap"]
F1=PRICE+["f1","f5","f20","fv5mc","fv20mc"]
F2=PRICE+["f1","f5","f20","i1","i5","i20","smart5","smart20","cons1","cons5"]
F3=PRICE+["f1","f5","f20","f_accel1","f_accel5","i_accel1","i_accel5"]
F4=PRICE+["f1","f5","i1","i5","absorb1","absorb5","cons1","cons5"]
F5=F2+["f_accel1","f_accel5","i_accel1","i_accel5","sec_pos1","sec_mean1","sec_disp1","sec_best1","sec_pos1m","sec_excess1m","sec_best_excess1m"]

def model():
    return HistGradientBoostingClassifier(max_depth=3,learning_rate=.045,max_iter=160,l2_regularization=8,min_samples_leaf=20,random_state=SEED)

# expanding walk-forward folds; last test date leaves 5 trading days of future.
folds=[
 ("2026-08-17","2026-08-18","2026-08-21"),
 ("2026-08-24","2026-08-24","2026-08-28"),
 ("2026-08-31","2026-08-31","2026-09-04"),
]
models={"BASE_PRICE":PRICE,"F1_FOREIGN_ACCUM":F1,"F2_SMARTMONEY_CONSENSUS":F2,
        "F3_FLOW_ACCEL":F3,"F4_ABSORPTION":F4,"F5_FLOW_SECTOR":F5}
pool={k:[] for k in models}

for cutoff,ts,te in folds:
    tr=x[(x.Date<=pd.Timestamp(cutoff)) & (x.elig==1)].copy()
    test=x[(x.Date>=pd.Timestamp(ts)) & (x.Date<=pd.Timestamp(te))].copy()
    if len(tr)<100 or len(test)==0: continue
    for name,feat in models.items():
        m=model(); m.fit(tr[feat],tr.W3)
        z=test.copy(); z["score"]=m.predict_proba(z[feat])[:,1]
        # live-like: rank at D0 close, at next open skip gap-ineligible; retain top10 + top2
        for dt,g in z.groupby("Date"):
            ge=g[g.elig==1].sort_values("score",ascending=False)
            for rankn,row in enumerate(ge.head(10).itertuples(),start=1):
                pool[name].append({"Date":dt,"Code":row.Code,"rank":rankn,"W3":int(row.W3),"pnl":float(row.pW3),"score":float(row.score)})

def summarize(rows,topn):
    z=pd.DataFrame(rows)
    if z.empty:return {}
    z=z[z["rank"]<=topn].sort_values(["Date","rank"])
    # 5-session stock dedupe in pooled forward predictions
    dates=sorted(z.Date.unique()); dpos={pd.Timestamp(d):i for i,d in enumerate(dates)}
    keep=[]; last={}
    for r in z.itertuples():
        p=dpos[pd.Timestamp(r.Date)]
        if r.Code in last and p-last[r.Code]<=4: continue
        last[r.Code]=p; keep.append(r.Index)
    z=z.loc[keep]
    n=len(z); k=int(z.W3.sum())
    byfold=[]
    for a,b in [("2026-08-18","2026-08-21"),("2026-08-24","2026-08-28"),("2026-08-31","2026-09-04")]:
        q=z[(z.Date>=pd.Timestamp(a))&(z.Date<=pd.Timestamp(b))]
        byfold.append({"period":a+"~"+b,"n":len(q),"wr":round(100*q.W3.mean(),2) if len(q) else None,"avg":round(q.pnl.mean(),3) if len(q) else None})
    return {"n":n,"k":k,"wr":round(100*k/n,2) if n else None,"avg":round(float(z.pnl.mean()),3) if n else None,
            "tail":round(100*float((z.pnl<=-5).mean()),2) if n else None,"folds":byfold}

results=[]
for name,rows in pool.items():
    results.append({"model":name,"top2":summarize(rows,2),"top10":summarize(rows,10)})
res={"study":"RECENT_FLOW_FIVE_MODELS_V1",
     "source":"K-Export Stars Korean Market Data (CC BY 4.0), archives 2026-07-31 onward; investor flows derived from KIS OpenAPI, sector indices from KRX",
     "coverage":{"snapshots":int(flows.Date.nunique()),"start":str(flows.Date.min().date()),"end":str(flows.Date.max().date()),"joined_rows":len(x),"joined_dates":int(x.Date.nunique())},
     "validation":"expanding blocked walk-forward; test blocks 2026-08-18..21, 08-24..28, 08-31..09-04; no later outcomes used in fitting",
     "trade_rule":"D0 close rank; D+1 open only if gap -3..+3%; TP +3% intraday before close-stop -3%; max5d; estimated cost 0.30%",
     "results":results}
print("FINAL_RESULT",json.dumps(res,ensure_ascii=False),flush=True)
open("recent_flow_five_models_v1_result.json","w").write(json.dumps(res,ensure_ascii=False,indent=2))
