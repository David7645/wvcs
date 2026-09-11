import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

SEED=42
COLS=["Date","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Stocks","Market"]
TARGETS=[("W3",3,5),("W4",4,7),("W5",5,10)]

F=[]
for y in range(2015,2027):
    x=pd.read_parquet(f"data/marcap-{y}.parquet",columns=COLS)
    F.append(x[x.Market.astype(str).str.upper().eq("KOSDAQ")])
d=pd.concat(F,ignore_index=True)
d["Date"]=pd.to_datetime(d.Date); d["Code"]=d.Code.astype(str).str.zfill(6)
for c in COLS[3:-1]: d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna()
d=d[(d.Open>0)&(d.High>0)&(d.Low>0)&(d.Close>0)&(d.Amount>0)&(d.Stocks>0)]
d=d[~d.Name.astype(str).str.contains("스팩",na=False)].sort_values(["Code","Date"]).reset_index(drop=True)
d["r1"]=d.groupby("Code").Close.pct_change()*100
d["lu"]=(d.ChangesRatio>=29.5)&(d.Close==d.High)
d["ap"]=d.groupby("Date").Amount.rank(pct=True,ascending=False)
d["mp"]=d.groupby("Date").Marcap.rank(pct=True,ascending=False)
d["r20tmp"]=d.groupby("Code").Close.pct_change(20)*100
d["mom_bucket"]=pd.qcut(d["r20tmp"].rank(method="first"),10,labels=False,duplicates="drop")
# datewise deciles for dynamic peer group
d["mdec"]=d.groupby("Date").Marcap.transform(lambda s: pd.qcut(s.rank(method="first"),10,labels=False,duplicates="drop"))
d["momdec"]=d.groupby("Date").r20tmp.transform(lambda s: pd.qcut(s.rank(method="first"),10,labels=False,duplicates="drop"))

# market factors
g=d.groupby("Date")
m=g.agg(eq=("ChangesRatio","mean"), breadth=("ChangesRatio",lambda s:(s>0).mean()),
        up5=("ChangesRatio",lambda s:(s>=5).mean()), amount=("Amount","sum"), lucount=("lu","sum"))
m["eq5"]=m["eq"].rolling(5,min_periods=3).mean()
m["eq20"]=m["eq"].rolling(20,min_periods=5).mean()
m["b5"]=m.breadth.rolling(5,min_periods=3).mean(); m["b20"]=m.breadth.rolling(20,min_periods=5).mean()
m["u5"]=m.up5.rolling(5,min_periods=3).mean(); m["u20"]=m.up5.rolling(20,min_periods=5).mean()
m["a20"]=m.amount.rolling(20,min_periods=5).median()
m["risk"]=(m.eq5.rank(pct=True)+m.b5.rank(pct=True)+m.u5.rank(pct=True)+(m.amount/m.a20).rank(pct=True))/4

leader=d[d.ChangesRatio>=5].groupby("Date").ChangesRatio.mean().rename("leader")
m=m.join(leader,how="left"); m["leader"]=m.leader.fillna(0)
# dynamic peer ecology by date x mcap decile x momentum decile
peer=d.groupby(["Date","mdec","momdec"]).agg(peer_ret=("ChangesRatio","mean"),peer_up5=("ChangesRatio",lambda s:(s>=5).mean()),peer_n=("Code","size")).reset_index()
peer_key={(pd.Timestamp(r.Date),int(r.mdec),int(r.momdec)):(float(r.peer_ret),float(r.peer_up5),int(r.peer_n)) for r in peer.itertuples() if pd.notna(r.mdec) and pd.notna(r.momdec)}
MD=m.to_dict("index")

rows=[]
for code,z in d.groupby("Code",sort=False):
    z=z.reset_index(drop=True)
    if len(z)<80: continue
    O,H,L,C,V,A,S,MC=z.Open.values,z.High.values,z.Low.values,z.Close.values,z.Volume.values,z.Amount.values,z.Stocks.values,z.Marcap.values
    R,LU,AP,MP=z.r1.values,z.lu.values,z.ap.values,z.mp.values
    D=z.Date.values
    mdec=z.mdec.values; momdec=z.momdec.values
    market=np.array([MD.get(pd.Timestamp(q),{}).get("eq",0) for q in D],float)
    leaderf=np.array([MD.get(pd.Timestamp(q),{}).get("leader",0) for q in D],float)
    for i in range(60,len(z)-11):
        dt=pd.Timestamp(D[i]); mm=MD.get(dt)
        if mm is None or not np.isfinite(mm.get("risk",np.nan)): continue
        if LU[i] or LU[i-20:i].any() or C[i]<1000 or AP[i]>.15 or A[i]<1e9 or not(-10<=R[i]<=20): continue
        if not np.isfinite(R[i-60:i+1]).all(): continue
        medv=np.median(V[i-20:i]); meda=np.median(A[i-20:i])
        if medv<=0 or meda<=0: continue

        gap=(O[i+1]/C[i]-1)*100; elig=int(-3<=gap<=3)
        out={}
        for nm,tp,hor in TARGETS:
            win=0; pnl=np.nan
            if elig:
                e=O[i+1]; gross=None
                for j in range(i+1,min(i+hor,len(z)-1)+1):
                    if H[j]>=e*(1+tp/100): win=1; gross=tp; break
                    if C[j]<=e*.97: gross=(C[j]/e-1)*100; break
                if gross is None: gross=(C[min(i+hor,len(z)-1)]/e-1)*100
                pnl=gross-.30
            out[nm]=win; out["p"+nm]=pnl

        # common trend/liquidity
        r5=(C[i]/C[i-5]-1)*100; r10=(C[i]/C[i-10]-1)*100; r20=(C[i]/C[i-20]-1)*100; r60=(C[i]/C[i-60]-1)*100
        rv5=np.std(R[i-4:i+1],ddof=1); rv20=np.std(R[i-19:i+1],ddof=1)
        vr=V[i]/medv; ar=A[i]/meda
        ma20=np.mean(C[i-19:i+1]); hi20=np.max(H[i-20:i])
        dayrange=max(H[i]-L[i],1e-9)
        upper=(H[i]-max(O[i],C[i]))/dayrange
        lower=(min(O[i],C[i])-L[i])/dayrange
        body=abs(C[i]-O[i])/dayrange
        clv=(C[i]-L[i])/dayrange

        # supply/dilution proxies from listed shares
        s1=(S[i]/S[i-1]-1)*100 if S[i-1]>0 else 0
        s5=(S[i]/S[i-5]-1)*100 if S[i-5]>0 else 0
        s20=(S[i]/S[i-20]-1)*100 if S[i-20]>0 else 0
        s60=(S[i]/S[i-60]-1)*100 if S[i-60]>0 else 0
        jumps=np.diff(S[i-60:i+1])/np.maximum(S[i-60:i],1)
        issue_count=int(np.sum(jumps>0.005)); max_issue=float(np.max(jumps)*100) if len(jumps) else 0

        # factor-neutral / residual momentum
        aa=R[i-19:i+1]; mmkt=market[i-19:i+1]; ll=leaderf[i-19:i+1]
        beta_m=beta_l=0.; corr_l=0.
        if np.std(mmkt)>1e-9: beta_m=float(np.cov(aa,mmkt,ddof=1)[0,1]/np.var(mmkt,ddof=1))
        if np.std(ll)>1e-9:
            beta_l=float(np.cov(aa,ll,ddof=1)[0,1]/np.var(ll,ddof=1))
            corr_l=float(np.corrcoef(aa,ll)[0,1])
        resid5=r5-beta_m*float(np.sum(market[i-4:i+1]))
        resid20=r20-beta_m*float(np.sum(market[i-19:i+1]))

        # dynamic peer ecology
        pk=(dt,int(mdec[i]) if pd.notna(mdec[i]) else -1,int(momdec[i]) if pd.notna(momdec[i]) else -1)
        pret,pup,pn=peer_key.get(pk,(0.,0.,0))

        rec={"Code":code,"Date":dt,"year":dt.year,"bar":i,"elig":elig,"gap":gap,
             "r1":R[i],"r5":r5,"r10":r10,"r20":r20,"r60":r60,"rv5":rv5,"rv20":rv20,
             "vr":vr,"ar":ar,"turn":V[i]/S[i],"amc":A[i]/MC[i],"mp":MP[i],
             "ma20":C[i]/ma20-1,"h20":C[i]/hi20-1,
             "upper":upper,"lower":lower,"body":body,"clv":clv,
             "s1":s1,"s5":s5,"s20":s20,"s60":s60,"issue_count":issue_count,"max_issue":max_issue,
             "beta_m":beta_m,"beta_l":beta_l,"corr_l":corr_l,"resid5":resid5,"resid20":resid20,
             "peer_ret":pret,"peer_up5":pup,"peer_n":pn,
             "breadth":mm["breadth"],"bacc":mm["b5"]-mm["b20"],"diff":mm["u5"]-mm["u20"],
             "risk":mm["risk"],"lucount":mm["lucount"]}
        rec.update(out); rows.append(rec)

ev=pd.DataFrame(rows).reset_index(drop=True)
tr=ev[(ev.year>=2015)&(ev.year<=2018)]
ca=ev[ev.year==2019]
te=ev[(ev.year>=2020)&(ev.year<=2026)]

BASE=["r1","r5","r10","r20","r60","rv5","rv20","vr","ar","turn","amc","mp","ma20","h20","risk","diff","bacc"]
SUP=BASE+["s1","s5","s20","s60","issue_count","max_issue"]
RES=BASE+["beta_m","beta_l","corr_l","resid5","resid20"]
QUAL=BASE+["upper","lower","body","clv"]
PEER=BASE+["peer_ret","peer_up5","peer_n","corr_l","beta_l"]

def hgb():
    return HistGradientBoostingClassifier(max_depth=3,learning_rate=.04,max_iter=190,l2_regularization=9,min_samples_leaf=25,random_state=SEED)
def reg():
    return HistGradientBoostingRegressor(max_depth=3,learning_rate=.04,max_iter=190,l2_regularization=9,min_samples_leaf=25,random_state=SEED)
def logit():
    return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.18,class_weight="balanced",max_iter=6000,random_state=SEED))])
def wil(k,n):
    if n<1:return 0
    z=1.64485;p=k/n;den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def dedup(frame):
    ids=[]; last={}
    for ix,r in frame.sort_values(["Date","Code"]).iterrows():
        if r.Code in last and r.bar-last[r.Code]<=5: continue
        last[r.Code]=r.bar; ids.append(ix)
    return ev.loc[ids]

def pick(scores,target):
    best=None
    for q in [.70,.75,.80,.85,.875,.90,.925,.95,.96,.97]:
        th=float(np.quantile(scores,q))
        z=dedup(ca[(scores>=th)&(ca.elig==1)])
        if len(z)<20: continue
        k=int(z[target].sum()); wr=k/len(z); avg=float(z["p"+target].mean()); ok=wr>=.75 and avg>=1
        key=(ok,wil(k,len(z)),avg,len(z))
        if best is None or key>best[0]: best=(key,{"th":th,"n":len(z),"k":k,"wr":wr,"avg":avg,"ok":ok})
    return best[1]

def audit(scores,target,th):
    z=dedup(te[(scores>=th)&(te.elig==1)])
    n=len(z); k=int(z[target].sum()) if n else 0
    yrs=[]
    for y in range(2020,2027):
        a=z[z.year==y]; nn=len(a); kk=int(a[target].sum()) if nn else 0
        yrs.append({"year":y,"n":nn,"wr":round(100*kk/nn,2) if nn else None,"avg":round(float(a["p"+target].mean()),3) if nn else None})
    return {"n":n,"k":k,"wr":round(100*k/n,2) if n else None,"avg":round(float(z["p"+target].mean()),3) if n else None,
            "tail":round(100*float((z["p"+target]<=-5).mean()),2) if n else None,"yearly":yrs}

R=[]; SCORE={}

def run_model(name,features,custom=None):
    best=None
    for T in ["W3","W4","W5"]:
        x=tr[tr.elig==1]
        m=hgb(); m.fit(x[features],x[T])
        c=m.predict_proba(ca[features])[:,1]; t=m.predict_proba(te[features])[:,1]
        if custom is not None: c,t=custom(c,t,ca,te)
        g=pick(c,T); a=audit(t,T,g["th"]); key=(g["ok"],wil(g["k"],g["n"]),g["avg"])
        if best is None or key>best[0]: best=(key,T,g,a,c,t)
    r={"model":name,"target":best[1],"cal":best[2],"audit":best[3]}
    return r,(best[4],best[5])

r,s=run_model("M6_SUPPLY_INTEGRITY",SUP); R.append(r); SCORE["m6"]=s
r,s=run_model("M7_RESIDUAL_ALPHA",RES); R.append(r); SCORE["m7"]=s

def quality_gate(c,t,ca,te):
    # user-aligned: long upper wick and no-volume breakouts are penalized, not hard-future filtered.
    qc=np.where((ca.upper<=.35)&(ca.vr>=1.0)&(ca.clv>=.60),1.0,.45)
    qt=np.where((te.upper<=.35)&(te.vr>=1.0)&(te.clv>=.60),1.0,.45)
    return c*qc,t*qt
r,s=run_model("M8_QUALITY_BREAKOUT",QUAL,quality_gate); R.append(r); SCORE["m8"]=s
r,s=run_model("M9_PEER_ECOLOGY",PEER); R.append(r); SCORE["m9"]=s

# M10 multiview consensus + tail veto + expected-return utility.
def rank_against(base,x):
    s=np.sort(base)
    return np.array([(np.searchsorted(s,v,side="right")+.5)/(len(s)+1) for v in x])
cparts=[]; tparts=[]
for c,t in SCORE.values():
    cparts.append(rank_against(c,c)); tparts.append(rank_against(c,t))
cons_c=np.prod(np.vstack(cparts),axis=0)**(1/4)
cons_t=np.prod(np.vstack(tparts),axis=0)**(1/4)
best=None
for T in ["W3","W4","W5"]:
    x=tr[tr.elig==1].copy()
    # tail classifier and PnL regressor use only train period
    x["TAIL"]=(x["p"+T]<=-5).astype(int)
    mt=hgb(); mt.fit(x[BASE+["upper","s20","resid20","peer_up5"]],x.TAIL)
    mr=reg(); mr.fit(x[BASE+["upper","s20","resid20","peer_up5"]],x["p"+T])
    tc=mt.predict_proba(ca[BASE+["upper","s20","resid20","peer_up5"]])[:,1]
    tt=mt.predict_proba(te[BASE+["upper","s20","resid20","peer_up5"]])[:,1]
    muc=mr.predict(ca[BASE+["upper","s20","resid20","peer_up5"]])
    mut=mr.predict(te[BASE+["upper","s20","resid20","peer_up5"]])
    c=cons_c*(1-tc)*(1/(1+np.exp(-muc/1.5)))
    t=cons_t*(1-tt)*(1/(1+np.exp(-mut/1.5)))
    g=pick(c,T); a=audit(t,T,g["th"]); key=(g["ok"],wil(g["k"],g["n"]),g["avg"])
    if best is None or key>best[0]: best=(key,T,g,a)
R.append({"model":"M10_MULTIVIEW_SAFETY","target":best[1],"cal":best[2],"audit":best[3]})

for r in R:
    a=r["audit"]; yrs=[y for y in a["yearly"] if y["n"]>0]
    r["pass"]=bool(a["n"]>=100 and a["wr"]>=75 and a["avg"]>=1 and a["tail"]<=10 and len(yrs)>=5 and min(y["wr"] for y in yrs)>=65 and min(y["avg"] for y in yrs)>=0)
P=[r for r in R if r["pass"]]
res={"study":"DAILYPLUS_FIVE_MODELS_V2","counts":{"all":len(ev),"train":len(tr),"cal":len(ca),"audit":len(te)},
     "models":["Supply Integrity","Residual Alpha","Quality Breakout","Peer Ecology","Multiview Safety"],
     "rule":"D0 close signal; D+1 open only if gap -3..+3%; close-stop -3%; cost 0.30%",
     "goal":{"wr":75,"avg_net":1.0,"n":100,"tail_max":10},
     "results":R,
     "champion":max(P,key=lambda r:(r["audit"]["wr"],r["audit"]["avg"],r["audit"]["n"])) if P else None}
print("FINAL_RESULT",json.dumps(res,ensure_ascii=False),flush=True)
open("dailyplus_five_models_v2_result.json","w").write(json.dumps(res,ensure_ascii=False,indent=2))
