import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
SEED=42
COLS=["Date","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Stocks","Market"]
F=[]
for y in range(2015,2027):
    x=pd.read_parquet(f"data/marcap-{y}.parquet",columns=COLS)
    F.append(x[x.Market.astype(str).str.upper().eq("KOSDAQ")])
d=pd.concat(F,ignore_index=True); d.Date=pd.to_datetime(d.Date); d.Code=d.Code.astype(str).str.zfill(6)
for c in COLS[3:-1]: d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna(); d=d[(d.Open>0)&(d.Close>0)&(d.Amount>0)&(d.Stocks>0)]
d=d[~d.Name.astype(str).str.contains("스팩",na=False)].sort_values(["Code","Date"]).reset_index(drop=True)
d["r1"]=d.groupby("Code").Close.pct_change()*100; d["lu"]=(d.ChangesRatio>=29.5)&(d.Close==d.High)
d["ap"]=d.groupby("Date").Amount.rank(pct=True,ascending=False)
g=d.groupby("Date")
m=g.agg(breadth=("ChangesRatio",lambda s:(s>0).mean()),up5=("ChangesRatio",lambda s:(s>=5).mean()),
        eq=("ChangesRatio","mean"),lucount=("lu","sum"),amt=("Amount","sum"))
m["b5"]=m.breadth.rolling(5,min_periods=3).mean(); m["b20"]=m.breadth.rolling(20,min_periods=5).mean()
m["u5"]=m.up5.rolling(5,min_periods=3).mean(); m["u20"]=m.up5.rolling(20,min_periods=5).mean()
m["e5"]=m["eq"].rolling(5,min_periods=3).mean(); m["a20"]=m.amt.rolling(20,min_periods=5).median()
m["risk"]=(m.e5.rank(pct=True)+m.b5.rank(pct=True)+m.u5.rank(pct=True)+(m.amt/m.a20).rank(pct=True))/4
m["leader"]=d[d.ChangesRatio>=5].groupby("Date").ChangesRatio.mean()
m["leader"]=m.leader.fillna(0); MD=m.to_dict("index")
rows=[]
for code,z in d.groupby("Code",sort=False):
    z=z.reset_index(drop=True)
    if len(z)<75: continue
    O,H,L,C,V,A,S=z.Open.values,z.High.values,z.Low.values,z.Close.values,z.Volume.values,z.Amount.values,z.Stocks.values
    R,LU,AP=z.r1.values,z.lu.values,z.ap.values; D=z.Date.values; N=z.Name.astype(str).values
    LF=np.array([MD.get(pd.Timestamp(q),{}).get("leader",0) for q in D],float)
    for i in range(60,len(z)-11):
        dt=pd.Timestamp(D[i]); mm=MD.get(dt)
        if mm is None or not np.isfinite(mm.get("risk",np.nan)): continue
        if LU[i] or LU[i-20:i].any() or C[i]<1000 or AP[i]>.15 or A[i]<1e9 or not(-10<=R[i]<=20): continue
        if not np.isfinite(R[i-60:i+1]).all(): continue
        medv=np.median(V[i-20:i]); meda=np.median(A[i-20:i])
        if medv<=0 or meda<=0: continue
        # sequence signature: six 10-day return/volume blocks
        sr=[]; sv=[]
        for a,b in [(-60,-50),(-50,-40),(-40,-30),(-30,-20),(-20,-10),(-10,0)]:
            sr.append((C[i+b-1]/C[i+a]-1)*100)
            sv.append(float(np.mean(V[i+a:i+b]/medv)))
        r5=(C[i]/C[i-5]-1)*100; r10=(C[i]/C[i-10]-1)*100; r20=(C[i]/C[i-20]-1)*100; r60=(C[i]/C[i-60]-1)*100
        rv5=np.std(R[i-4:i+1],ddof=1); rv20=np.std(R[i-19:i+1],ddof=1)
        hi20=np.max(H[i-20:i]); ma20=np.mean(C[i-19:i+1])
        corr=0.
        aa=R[i-19:i+1]; bb=LF[i-19:i+1]
        if np.std(aa)>1e-9 and np.std(bb)>1e-9: corr=float(np.corrcoef(aa,bb)[0,1])
        prelu=int(LU[i+1:i+6].any())
        daylu=int(np.flatnonzero(LU[i+1:i+6])[0]+1) if prelu else 0
        gap=(O[i+1]/C[i]-1)*100; elig=int(-3<=gap<=3)
        out={}
        for nm,tp,hor in [("W3",3,5),("W4",4,7),("W5",5,10)]:
            win=0; pnl=np.nan
            if elig:
                e=O[i+1]; gross=None
                for j in range(i+1,min(i+hor,len(z)-1)+1):
                    if H[j]>=e*(1+tp/100): win=1; gross=tp; break
                    if C[j]<=e*.97: gross=(C[j]/e-1)*100; break
                if gross is None: gross=(C[min(i+hor,len(z)-1)]/e-1)*100
                pnl=gross-.30
            out[nm]=win; out["p"+nm]=pnl
        rec={"Code":code,"Date":dt,"year":dt.year,"bar":i,"PRELU":prelu,"daylu":daylu,"elig":elig,"gap":gap,
             "r1":R[i],"r5":r5,"r10":r10,"r20":r20,"r60":r60,"rv5":rv5,"rv20":rv20,"comp":rv5/max(rv20,1e-6),
             "vr":V[i]/medv,"ar":A[i]/meda,"turn":V[i]/S[i],"h20":C[i]/hi20-1,"ma20":C[i]/ma20-1,
             "corr":corr,"breadth":mm["breadth"],"bacc":mm["b5"]-mm["b20"],"diff":mm["u5"]-mm["u20"],
             "risk":mm["risk"],"lucount":mm["lucount"]}
        rec.update({f"sr{k}":v for k,v in enumerate(sr)}); rec.update({f"sv{k}":v for k,v in enumerate(sv)}); rec.update(out); rows.append(rec)
ev=pd.DataFrame(rows).reset_index(drop=True)
tr=ev[(ev.year>=2015)&(ev.year<=2018)]; ca=ev[ev.year==2019]; te=ev[(ev.year>=2020)&(ev.year<=2026)]
SEQ=[f"sr{k}" for k in range(6)]+[f"sv{k}" for k in range(6)]+["r1","r5","r10","r20","r60","rv5","rv20","vr","ar","h20","ma20","risk"]
COMP=["r1","r5","r10","r20","rv5","rv20","comp","vr","ar","turn","h20","ma20","risk","diff"]
LEAD=COMP+["corr","breadth","bacc","lucount"]
def logit(): return Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.2,class_weight="balanced",max_iter=5000,random_state=SEED))])
def hgb(): return HistGradientBoostingClassifier(max_depth=3,learning_rate=.04,max_iter=180,l2_regularization=8,min_samples_leaf=25,random_state=SEED)
def wil(k,n):
    if n<1:return 0
    z=1.64485;p=k/n;den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den
def pick(scores,target):
    best=None
    for q in [.7,.75,.8,.85,.875,.9,.925,.95,.96,.97]:
        th=float(np.quantile(scores,q)); z=ca[(scores>=th)&(ca.elig==1)].sort_values(["Date","Code"])
        ids=[];last={}
        for ix,r in z.iterrows():
            if r.Code in last and r.bar-last[r.Code]<=5: continue
            last[r.Code]=r.bar;ids.append(ix)
        z=ev.loc[ids]
        if len(z)<20: continue
        k=int(z[target].sum()); wr=k/len(z); avg=float(z["p"+target].mean()); ok=wr>=.75 and avg>=1
        key=(ok,wil(k,len(z)),avg,len(z))
        if best is None or key>best[0]: best=(key,{"th":th,"n":len(z),"k":k,"wr":wr,"avg":avg,"ok":ok})
    return best[1]
def audit(scores,target,th):
    z=te[(scores>=th)&(te.elig==1)].sort_values(["Date","Code"]);ids=[];last={}
    for ix,r in z.iterrows():
        if r.Code in last and r.bar-last[r.Code]<=5: continue
        last[r.Code]=r.bar;ids.append(ix)
    z=ev.loc[ids];n=len(z);k=int(z[target].sum())
    yrs=[]
    for y in range(2020,2027):
        a=z[z.year==y]; nn=len(a); kk=int(a[target].sum()) if nn else 0
        yrs.append({"year":y,"n":nn,"wr":round(100*kk/nn,2) if nn else None,"avg":round(float(a["p"+target].mean()),3) if nn else None})
    return {"n":n,"k":k,"wr":round(100*k/n,2) if n else None,"avg":round(float(z["p"+target].mean()),3) if n else None,
            "tail":round(100*float((z["p"+target]<=-5).mean()),2) if n else None,"prelu":round(100*z.PRELU.mean(),2) if n else None,"yearly":yrs}
R=[]; SCORES={}
# M1 sequence
m=logit();m.fit(tr[SEQ],tr.PRELU); c=m.predict_proba(ca[SEQ])[:,1]; t=m.predict_proba(te[SEQ])[:,1]
best=None
for T in ["W3","W4","W5"]:
    g=pick(c,T);a=audit(t,T,g["th"]);key=(g["ok"],wil(g["k"],g["n"]),g["avg"])
    if best is None or key>best[0]:best=(key,T,g,a)
R.append({"model":"M1_PRELU_SEQUENCE","target":best[1],"cal":best[2],"audit":best[3]});SCORES["m1"]=(c,t)
# M2 compression
best=None
for T in ["W3","W4","W5"]:
    x=tr[tr.elig==1];m=hgb();m.fit(x[COMP],x[T]);c=m.predict_proba(ca[COMP])[:,1];t=m.predict_proba(te[COMP])[:,1]
    g=pick(c,T);a=audit(t,T,g["th"]);key=(g["ok"],wil(g["k"],g["n"]),g["avg"])
    if best is None or key>best[0]:best=(key,T,g,a,c,t)
R.append({"model":"M2_COMPRESSION_EXPANSION","target":best[1],"cal":best[2],"audit":best[3]});SCORES["m2"]=(best[4],best[5])
# M3 leader diffusion
best=None
for T in ["W3","W4","W5"]:
    x=tr[tr.elig==1];m=hgb();m.fit(x[LEAD],x[T]);c=m.predict_proba(ca[LEAD])[:,1];t=m.predict_proba(te[LEAD])[:,1]
    g=pick(c,T);a=audit(t,T,g["th"]);key=(g["ok"],wil(g["k"],g["n"]),g["avg"])
    if best is None or key>best[0]:best=(key,T,g,a,c,t)
R.append({"model":"M3_LEADER_DIFFUSION","target":best[1],"cal":best[2],"audit":best[3]});SCORES["m3"]=(best[4],best[5])
# M4 discrete hazard * direct win
HC=[];HT=[]
for h in range(1,6):
    y=(tr.daylu.values==h).astype(int);m=logit();m.fit(tr[LEAD],y);HC.append(m.predict_proba(ca[LEAD])[:,1]);HT.append(m.predict_proba(te[LEAD])[:,1])
sc=np.ones(len(ca));st=np.ones(len(te));cc=np.zeros(len(ca));ct=np.zeros(len(te))
for a,b in zip(HC,HT):cc+=sc*a;ct+=st*b;sc*=1-a;st*=1-b
best=None
for T in ["W3","W4","W5"]:
    x=tr[tr.elig==1];m=hgb();m.fit(x[COMP],x[T]);wc=m.predict_proba(ca[COMP])[:,1];wt=m.predict_proba(te[COMP])[:,1]
    c=np.sqrt(np.clip(cc*wc,1e-9,1));t=np.sqrt(np.clip(ct*wt,1e-9,1));g=pick(c,T);a=audit(t,T,g["th"]);key=(g["ok"],wil(g["k"],g["n"]),g["avg"])
    if best is None or key>best[0]:best=(key,T,g,a,c,t)
R.append({"model":"M4_PRELU_HAZARD","target":best[1],"cal":best[2],"audit":best[3]});SCORES["m4"]=(best[4],best[5])
# M5 two-stage consensus + next-open gap, trained on 2018 only, calibrated 2019
def rankv(base,x):
    s=np.sort(base);return np.array([(np.searchsorted(s,v,side="right")+.5)/(len(s)+1) for v in x])
cparts=[];tparts=[]
for c,t in SCORES.values(): cparts.append(rankv(c,c));tparts.append(rankv(c,t))
cons_c=np.prod(np.vstack(cparts),axis=0)**(1/4);cons_t=np.prod(np.vstack(tparts),axis=0)**(1/4)
# train compact second-stage on 2018 with fresh first-stage models fit <=2017
tr17=ev[(ev.year>=2015)&(ev.year<=2017)]; va18=ev[ev.year==2018]
a=logit();a.fit(tr17[SEQ],tr17.PRELU);p1=a.predict_proba(va18[SEQ])[:,1]
x=tr17[tr17.elig==1];b=hgb();b.fit(x[COMP],x.W4);p2=b.predict_proba(va18[COMP])[:,1]
c0=hgb();c0.fit(x[LEAD],x.W4);p3=c0.predict_proba(va18[LEAD])[:,1]
cons18=(p1*p2*p3)**(1/3)
best=None
for T in ["W3","W4","W5"]:
    v=va18[va18.elig==1].copy(); X=np.c_[cons18[va18.elig.values==1],v.gap.values,v.r1.values,v.r5.values,v.comp.values,v.ar.values,v["corr"].values,v.risk.values]
    sm=hgb();sm.fit(X,v[T])
    Xc=np.c_[cons_c,ca.gap.values,ca.r1.values,ca.r5.values,ca.comp.values,ca.ar.values,ca["corr"].values,ca.risk.values]
    Xt=np.c_[cons_t,te.gap.values,te.r1.values,te.r5.values,te.comp.values,te.ar.values,te["corr"].values,te.risk.values]
    cs=sm.predict_proba(Xc)[:,1];ts=sm.predict_proba(Xt)[:,1];g=pick(cs,T);au=audit(ts,T,g["th"]);key=(g["ok"],wil(g["k"],g["n"]),g["avg"])
    if best is None or key>best[0]:best=(key,T,g,au)
R.append({"model":"M5_TWO_STAGE_ALPHA","target":best[1],"cal":best[2],"audit":best[3]})
for r in R:
    a=r["audit"];yrs=[y for y in a["yearly"] if y["n"]>0]
    r["pass"]=bool(a["n"]>=100 and a["wr"]>=75 and a["avg"]>=1 and a["tail"]<=10 and len(yrs)>=5 and min(y["wr"] for y in yrs)>=65 and min(y["avg"] for y in yrs)>=0)
P=[r for r in R if r["pass"]]
res={"study":"PRELU_FIVE_MODELS_V1","counts":{"all":len(ev),"prelu":int(ev.PRELU.sum()),"train":len(tr),"cal":len(ca),"audit":len(te)},
     "rule":"D0 close signal; D+1 open only if gap -3..+3%; close-stop -3%; cost 0.30%",
     "goal":{"wr":75,"avg_net":1.0,"n":100,"tail_max":10},"results":R,
     "champion":max(P,key=lambda r:(r["audit"]["wr"],r["audit"]["avg"],r["audit"]["n"])) if P else None}
print("FINAL_RESULT",json.dumps(res,ensure_ascii=False),flush=True)
open("prelu_five_models_v1_result.json","w").write(json.dumps(res,ensure_ascii=False,indent=2))
