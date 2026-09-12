import json, math, time, warnings, threading
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

SEED=42
H={"User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
   "Referer":"https://m.stock.naver.com/","Accept":"application/json"}

# ---------------- market bars ----------------
cols=["Date","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Stocks","Market"]
d=pd.read_parquet("data/marcap-2026.parquet",columns=cols)
d=d[d.Market.astype(str).str.upper().eq("KOSDAQ")].copy()
d.Date=pd.to_datetime(d.Date); d.Code=d.Code.astype(str).str.zfill(6)
for c in cols[3:-1]: d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna()
d=d[(d.Open>0)&(d.High>0)&(d.Low>0)&(d.Close>0)&(d.Amount>0)&(d.Stocks>0)]
d=d[~d.Name.astype(str).str.contains("스팩",na=False)].sort_values(["Code","Date"]).reset_index(drop=True)
d["r1"]=d.groupby("Code").Close.pct_change()*100
d["r5"]=d.groupby("Code").Close.pct_change(5)*100
d["r10"]=d.groupby("Code").Close.pct_change(10)*100
d["r20"]=d.groupby("Code").Close.pct_change(20)*100
d["rv5"]=d.groupby("Code").r1.rolling(5).std().reset_index(level=0,drop=True)
d["rv20"]=d.groupby("Code").r1.rolling(20).std().reset_index(level=0,drop=True)
d["vmed20"]=d.groupby("Code").Volume.rolling(20).median().reset_index(level=0,drop=True)
d["amed20"]=d.groupby("Code").Amount.rolling(20).median().reset_index(level=0,drop=True)
d["vr"]=d.Volume/d.vmed20; d["ar"]=d.Amount/d.amed20
d["ap"]=d.groupby("Date").Amount.rank(pct=True,ascending=False)
d["lu"]=(d.ChangesRatio>=29.5)&(d.Close==d.High)
d["range"]=np.maximum(d.High-d.Low,1e-9)
d["upper"]=(d.High-np.maximum(d.Open,d.Close))/d["range"]
d["clv"]=(d.Close-d.Low)/d["range"]

# Limit to names that are repeatedly liquid in recent 70 sessions.
last_dates=sorted(d.Date.unique())[-75:]
recent=d[d.Date.isin(last_dates)].copy()
eligible_codes=(recent[(recent.ap<=.30)&(recent.Close>=1000)&(recent.Amount>=1e9)]
                .groupby("Code").size())
codes=eligible_codes[eligible_codes>=3].index.tolist()
print("FETCH_CODES",len(codes),flush=True)

# ---------------- fetch Naver 60d investor trend ----------------
_tls=threading.local()
def sess():
    if not hasattr(_tls,"s"):
        _tls.s=requests.Session(); _tls.s.headers.update(H)
    return _tls.s

def fetch(code):
    url=f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize=60"
    for k in range(3):
        try:
            r=sess().get(url,timeout=10)
            if r.status_code==200:
                arr=r.json()
                if not isinstance(arr,list): arr=arr.get("list") or arr.get("result") or []
                out=[]
                for z in arr:
                    dt=str(z.get("bizdate") or "")
                    if len(dt)!=8: continue
                    def n(key):
                        v=z.get(key,0)
                        try:return float(str(v).replace(",",""))
                        except:return 0.0
                    out.append({"Code":code,"Date":pd.to_datetime(dt,format="%Y%m%d"),
                                "f":n("foreignerPureBuyQuant"),"i":n("organPureBuyQuant"),
                                "p":n("individualPureBuyQuant"),
                                "fr":n("foreignerHoldRatio"),"close_n":n("closePrice")})
                return out
            time.sleep(.25*(k+1))
        except Exception:
            time.sleep(.25*(k+1))
    return []

flowrows=[]
with ThreadPoolExecutor(max_workers=18) as ex:
    fut={ex.submit(fetch,c):c for c in codes}
    done=0
    for f in as_completed(fut):
        flowrows.extend(f.result()); done+=1
        if done%100==0: print("FETCHED",done,flush=True)
flows=pd.DataFrame(flowrows)
print("FLOW_ROWS",len(flows),"CODES",flows.Code.nunique() if len(flows) else 0,
      "DATES",flows.Date.min() if len(flows) else None,flows.Date.max() if len(flows) else None,flush=True)
if len(flows)==0: raise RuntimeError("no flow data")

# Rolling flow state per stock, strictly through D0.
flows=flows.sort_values(["Code","Date"]).reset_index(drop=True)
for c in ["f","i","p"]:
    for w in [3,5,10,20]:
        flows[f"{c}{w}"]=flows.groupby("Code")[c].rolling(w,min_periods=max(2,w//2)).sum().reset_index(level=0,drop=True)
flows["f_accel"]=flows.f3/3-flows.f10/10
flows["i_accel"]=flows.i3/3-flows.i10/10
flows["smart5"]=flows.f5+flows.i5
flows["smart20"]=flows.f20+flows.i20
flows["cons1"]=((flows.f>0)&(flows.i>0)).astype(int)
flows["cons5"]=((flows.f5>0)&(flows.i5>0)).astype(int)
flows["f_streak"]=flows.groupby("Code").f.transform(lambda s: s.gt(0).rolling(5,min_periods=1).sum())
flows["i_streak"]=flows.groupby("Code").i.transform(lambda s: s.gt(0).rolling(5,min_periods=1).sum())
flows["dispersion5"]=abs(flows.f5-flows.i5)

x=d.merge(flows,on=["Code","Date"],how="inner")
x=x[x.Code.isin(codes)].copy()
# normalize by shares outstanding
for c in ["f","i","p","f3","f5","f10","f20","i3","i5","i10","i20","p5","p20","smart5","smart20","dispersion5"]:
    if c in x: x[c+"_n"]=x[c]/x.Stocks
x["f_accel_n"]=x.f_accel/x.Stocks; x["i_accel_n"]=x.i_accel/x.Stocks

# Trade labels: D+1 open, gap +/-3, TP +3 / +4 / +5 before -3 close-stop; costs .30.
records=[]
for code,z in x.groupby("Code",sort=False):
    z=z.sort_values("Date").reset_index(drop=True)
    O,H,C=z.Open.to_numpy(float),z.High.to_numpy(float),z.Close.to_numpy(float)
    for k in range(len(z)):
        if k+10>=len(z): continue
        r=z.iloc[k].to_dict()
        gap=(O[k+1]/C[k]-1)*100; r["gap"]=gap; r["elig"]=int(-3<=gap<=3)
        for nm,tp,hor in [("W3",3,5),("W4",4,7),("W5",5,10)]:
            win=0; pnl=np.nan
            if r["elig"]:
                e=O[k+1]; gross=None
                for j in range(k+1,min(k+hor,len(z)-1)+1):
                    if H[j]>=e*(1+tp/100): win=1; gross=tp; break
                    if C[j]<=e*.97: gross=(C[j]/e-1)*100; break
                if gross is None: gross=(C[min(k+hor,len(z)-1)]/e-1)*100
                pnl=gross-.30
            r[nm]=win; r["p"+nm]=pnl
        records.append(r)
ev=pd.DataFrame(records)
ev=ev[(ev.Date<=pd.Timestamp("2026-09-04")) & (ev.Close>=1000) & (ev.Amount>=1e9) &
      (ev.ap<=.30) & (ev.lu==0) & (ev.upper<=.65)].replace([np.inf,-np.inf],np.nan)
print("EVENTS",len(ev),"DATES",ev.Date.min(),ev.Date.max(),flush=True)

PRICE=["r1","r5","r10","r20","rv5","rv20","vr","ar","ap","upper","clv"]
F_FOREIGN=PRICE+["f_n","f3_n","f5_n","f10_n","f20_n","f_accel_n","f_streak","fr"]
F_INST=PRICE+["i_n","i3_n","i5_n","i10_n","i20_n","i_accel_n","i_streak"]
F_CONS=PRICE+["f5_n","f20_n","i5_n","i20_n","smart5_n","smart20_n","cons1","cons5","dispersion5_n"]
F_CONTRA=PRICE+["f_n","i_n","p5_n","f_accel_n","i_accel_n","f_streak","i_streak"]
F_ALL=sorted(set(F_FOREIGN+F_INST+F_CONS+F_CONTRA))
ev=ev.dropna(subset=F_ALL+["W3","pW3"])

def clf():
    return HistGradientBoostingClassifier(max_depth=3,learning_rate=.04,max_iter=180,l2_regularization=8,min_samples_leaf=20,random_state=SEED)

# Four matured walk-forward blocks. Training cutoff leaves >=10 calendar days before test start.
folds=[
 ("2026-07-10","2026-07-20","2026-07-24"),
 ("2026-07-24","2026-08-03","2026-08-07"),
 ("2026-08-07","2026-08-17","2026-08-21"),
 ("2026-08-21","2026-08-31","2026-09-04"),
]
models={"M1_FOREIGN_PERSISTENCE":F_FOREIGN,
        "M2_INSTITUTION_PERSISTENCE":F_INST,
        "M3_SMARTMONEY_CONSENSUS":F_CONS,
        "M4_FLOW_CONTRARIAN":F_CONTRA,
        "M5_MULTIVIEW_FLOW":F_ALL}
preds={k:[] for k in models}
baseline=[]

for cutoff,ts,te in folds:
    tr=ev[(ev.Date<=pd.Timestamp(cutoff))&(ev.elig==1)].copy()
    tst=ev[(ev.Date>=pd.Timestamp(ts))&(ev.Date<=pd.Timestamp(te))].copy()
    print("FOLD",cutoff,ts,te,"TR",len(tr),"TEST",len(tst),flush=True)
    if len(tr)<500 or len(tst)==0: continue
    # baseline price model
    bm=clf(); bm.fit(tr[PRICE],tr.W3)
    zb=tst.copy(); zb["score"]=bm.predict_proba(zb[PRICE])[:,1]
    for dt,g in zb.groupby("Date"):
        ge=g[g.elig==1].sort_values("score",ascending=False)
        for rank,row in enumerate(ge.head(10).itertuples(),1):
            baseline.append({"Date":dt,"Code":row.Code,"rank":rank,"W3":int(row.W3),"pnl":float(row.pW3)})
    for name,feat in models.items():
        m=clf();m.fit(tr[feat],tr.W3)
        z=tst.copy();z["score"]=m.predict_proba(z[feat])[:,1]
        # M5 adds explicit no-long-wick/no-empty-breakout guard
        if name=="M5_MULTIVIEW_FLOW":
            z["score"]*=np.where((z.upper<=.35)&(z.vr>=1.0),1.0,.55)
        for dt,g in z.groupby("Date"):
            ge=g[g.elig==1].sort_values("score",ascending=False)
            for rank,row in enumerate(ge.head(10).itertuples(),1):
                preds[name].append({"Date":dt,"Code":row.Code,"rank":rank,"W3":int(row.W3),"pnl":float(row.pW3)})

def summ(rows,n):
    z=pd.DataFrame(rows)
    if z.empty:return {}
    z=z[z["rank"]<=n].copy()
    # stock repeat dedupe: 5 trading sessions using actual selected date order
    dates=sorted(z.Date.unique()); pos={pd.Timestamp(v):j for j,v in enumerate(dates)}
    keep=[];last={}
    for r in z.sort_values(["Date","rank"]).itertuples():
        p=pos[pd.Timestamp(r.Date)]
        if r.Code in last and p-last[r.Code]<=4: continue
        last[r.Code]=p;keep.append(r.Index)
    z=z.loc[keep]
    N=len(z);K=int(z.W3.sum())
    return {"n":N,"k":K,"wr":round(100*K/N,2) if N else None,
            "avg":round(float(z.pnl.mean()),3) if N else None,
            "tail":round(100*float((z.pnl<=-5).mean()),2) if N else None}

res=[{"model":"BASE_PRICE","top2":summ(baseline,2),"top5":summ(baseline,5),"top10":summ(baseline,10)}]
for name,rows in preds.items():
    res.append({"model":name,"top2":summ(rows,2),"top5":summ(rows,5),"top10":summ(rows,10)})
out={"study":"NAVER60_FLOW_FIVE_MODELS_V2",
     "coverage":{"flow_codes":int(flows.Code.nunique()),"flow_rows":len(flows),"flow_start":str(flows.Date.min().date()),"flow_end":str(flows.Date.max().date()),"event_rows":len(ev)},
     "validation":"4 blocked walk-forward folds; training labels fully matured before each test block",
     "rule":"D0 close rank; D+1 open only gap -3..+3%; TP +3%; close-stop -3%; max5d; cost .30%",
     "results":res}
print("FINAL_RESULT",json.dumps(out,ensure_ascii=False),flush=True)
flows.to_parquet("naver60_flow_cache.parquet",index=False)
open("naver60_flow_five_models_v2_result.json","w").write(json.dumps(out,ensure_ascii=False,indent=2))
