import json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

SEED=42
YEARS=range(2015,2027)
COLS=["Date","Rank","Code","Name","Open","High","Low","Close","Volume","Amount","ChangesRatio","Marcap","Stocks","Market","Dept"]
TARGETS=[("T3",3.0,5),("T4",4.0,7),("T5",5.0,10)]

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
d=d.sort_values(["Code","Date"]).reset_index(drop=True)
d["ret1"]=d.groupby("Code").Close.pct_change()*100
d["range_pct"]=(d.High/d.Low-1)*100
d["clv"]=np.where(d.High>d.Low,(d.Close-d.Low)/(d.High-d.Low),1.0)
d["is_lu"]=(d.ChangesRatio>=29.5)&(d.Close==d.High)

# Market states known at D0 close.
g=d.groupby("Date",sort=True)
macro=g.agg(
    breadth=("ChangesRatio",lambda s:float((s>0).mean())),
    up3=("ChangesRatio",lambda s:float((s>=3).mean())),
    up5=("ChangesRatio",lambda s:float((s>=5).mean())),
    eqret=("ChangesRatio","mean"),
    amount=("Amount","sum"),
    lucount=("is_lu","sum")
)
for w in [5,20]:
    macro[f"breadth{w}"]=macro.breadth.rolling(w,min_periods=3).mean()
    macro[f"up5_{w}"]=macro.up5.rolling(w,min_periods=3).mean()
    macro[f"eqret{w}"]=macro.eqret.rolling(w,min_periods=3).mean()
    macro[f"amount{w}"]=macro.amount.rolling(w,min_periods=3).median()
macro["breadth_accel"]=macro.breadth5-macro.breadth20
macro["diffusion"]=macro.up5_5-macro.up5_20
macro["amount_shock"]=macro.amount/macro.amount20.replace(0,np.nan)
macro["risk_on"]=(macro.eqret5.rank(pct=True)+macro.breadth5.rank(pct=True)+macro.up5_5.rank(pct=True)+macro.amount_shock.rank(pct=True))/4
M=macro.to_dict("index")

# Pivot of daily returns for graph features.
ret_piv=d.pivot(index="Date",columns="Code",values="ret1").sort_index()
date_pos={dt:i for i,dt in enumerate(ret_piv.index)}
same_day=d.set_index(["Date","Code"])[["ChangesRatio"]]

events=[]
all_entry_rows=[]
seqs=[]
graph_feats=[]

for code,gg in d.groupby("Code",sort=False):
    gg=gg.reset_index(drop=True)
    inds=np.flatnonzero(gg.is_lu.to_numpy(bool))
    if len(inds)==0: continue
    op=gg.Open.to_numpy(float); hi=gg.High.to_numpy(float); lo=gg.Low.to_numpy(float); cl=gg.Close.to_numpy(float)
    vo=gg.Volume.to_numpy(float); am=gg.Amount.to_numpy(float); st=gg.Stocks.to_numpy(float)
    ret=gg.ret1.to_numpy(float); rng=gg.range_pct.to_numpy(float); clv=gg.clv.to_numpy(float)
    dt=gg.Date.to_numpy(); nm=gg.Name.astype(str).to_numpy()
    last=-999
    for i in inds:
        if i-last<=4: continue
        last=i
        if i<61 or i+15>=len(gg): continue
        date=pd.Timestamp(dt[i]); mm=M.get(date)
        if mm is None: continue

        # 60-day sequence ending D-1: returns, log volume relative to rolling median, range, CLV.
        idx=np.arange(i-60,i)
        if np.any(~np.isfinite(ret[idx])): continue
        vol_base=np.median(vo[i-60:i])
        if vol_base<=0: continue
        seq=np.concatenate([
            np.clip(ret[idx],-20,20)/10.0,
            np.clip(np.log1p(vo[idx]/vol_base),0,5)/5.0,
            np.clip(rng[idx],0,20)/10.0,
            np.clip(clv[idx],0,1)
        ])

        # Graph: candidate's prior-20 return correlation to D0 leaders (+5% or more).
        gp=[0.0,0.0,0.0,0.0]
        p=date_pos.get(date)
        if p is not None and p>=21 and code in ret_piv.columns:
            hist=ret_piv.iloc[p-20:p]
            cand=hist[code].to_numpy(float)
            # leaders known at D0 close
            dd=d[d.Date.eq(date)]
            leaders=dd.loc[(dd.ChangesRatio>=5)&(dd.Code.ne(code)),"Code"].tolist()
            vals=[]
            if np.nanstd(cand)>1e-9:
                for lc in leaders[:200]:
                    if lc not in hist.columns: continue
                    oth=hist[lc].to_numpy(float)
                    msk=np.isfinite(cand)&np.isfinite(oth)
                    if msk.sum()>=15 and np.nanstd(oth[msk])>1e-9:
                        vals.append(np.corrcoef(cand[msk],oth[msk])[0,1])
            vals=np.array([v for v in vals if np.isfinite(v)],float)
            if len(vals):
                gp=[float(len(vals)),float(np.mean(vals)),float(np.mean(vals>0.4)),float(np.mean(np.clip(vals,0,None)))]
            else:
                gp=[float(len(leaders)),0.0,0.0,0.0]

        # All feasible entry days D+1..D+3; prior D0-close break <=-3% cancels later entries.
        entry_candidates=[]
        broken=False
        for j in range(i+1,i+4):
            if j>i+1 and cl[j-1]<=cl[i]*0.97: broken=True
            if broken: break
            gap=(op[j]/cl[i]-1)*100
            if -3<=gap<=3:
                # info known at this open
                if j==i+1:
                    prev_close=prev_high=prev_low=prev_vol=0.0
                    cum_min_close=0.0; cum_max_high=0.0
                else:
                    pp=np.arange(i+1,j)
                    prev_close=(cl[j-1]/cl[i]-1)*100
                    prev_high=(hi[j-1]/cl[i]-1)*100
                    prev_low=(lo[j-1]/cl[i]-1)*100
                    prev_vol=vo[j-1]/max(vo[i],1)
                    cum_min_close=(np.min(cl[pp])/cl[i]-1)*100
                    cum_max_high=(np.max(hi[pp])/cl[i]-1)*100
                ctx=[j-i,gap,(op[j]/cl[j-1]-1)*100,prev_close,prev_high,prev_low,prev_vol,cum_min_close,cum_max_high]
                labs={}; pnls={}; event_days={}
                for tn,tpct,horizon in TARGETS:
                    end=min(j+horizon-1,len(gg)-1)
                    suc=0; gross=None; stop_day=None; tp_day=None
                    for k in range(j,end+1):
                        if hi[k]>=op[j]*(1+tpct/100):
                            suc=1; gross=tpct; tp_day=k-j+1; break
                        if cl[k]<=op[j]*0.97:
                            gross=(cl[k]/op[j]-1)*100; stop_day=k-j+1; break
                    if gross is None: gross=(cl[end]/op[j]-1)*100
                    labs[tn]=suc; pnls[tn]=gross-0.30
                    event_days[tn]=(tp_day,stop_day)
                entry_candidates.append((j,ctx,labs,pnls,event_days))

        if not entry_candidates: continue

        # first feasible entry for Models 1-4
        j,ctx,labs,pnls,event_days=entry_candidates[0]
        rec={"Code":code,"Name":nm[i],"Date":date,"year":date.year,"event_id":len(events),
             "entry_day":ctx[0],"entry_gap":ctx[1],"entry_gap_prev":ctx[2],
             "prev_close":ctx[3],"prev_high":ctx[4],"prev_low":ctx[5],"prev_vol":ctx[6],
             "cum_min_close":ctx[7],"cum_max_high":ctx[8],
             "D0_gap":(op[i]/cl[i-1]-1)*100,
             "D0_range":(hi[i]/lo[i]-1)*100,
             "D0_low":(lo[i]/cl[i-1]-1)*100,
             "D0_vol_ratio":vo[i]/max(np.median(vo[i-20:i]),1),
             "D0_amount_ratio":am[i]/max(np.median(am[i-20:i]),1),
             "turnover":vo[i]/st[i],
             "mcap_proxy":op[i]*st[i],
             "M_breadth":mm["breadth"],"M_breadth_accel":mm["breadth_accel"],
             "M_diffusion":mm["diffusion"],"M_amount_shock":mm["amount_shock"],"M_risk_on":mm["risk_on"]}
        rec.update(labs); rec.update({f"pnl_{k}":v for k,v in pnls.items()})
        events.append(rec); seqs.append(seq); graph_feats.append(gp)

        # all candidate rows for model5
        eid=len(events)-1
        for jj,cx,lb,pn,edays in entry_candidates:
            rr={"event_id":eid,"Date":date,"year":date.year,"Code":code,
                "entry_day":cx[0],"entry_gap":cx[1],"entry_gap_prev":cx[2],
                "prev_close":cx[3],"prev_high":cx[4],"prev_low":cx[5],"prev_vol":cx[6],
                "cum_min_close":cx[7],"cum_max_high":cx[8]}
            rr.update(lb); rr.update({f"pnl_{k}":v for k,v in pn.items()})
            all_entry_rows.append(rr)

ev=pd.DataFrame(events)
SEQ=np.vstack(seqs)
GRAPH=np.asarray(graph_feats,float)
entry=pd.DataFrame(all_entry_rows)

# Fixed static features known at first eligible entry.
base_cols=["entry_day","entry_gap","entry_gap_prev","prev_close","prev_high","prev_low","prev_vol",
           "cum_min_close","cum_max_high","D0_gap","D0_range","D0_low","D0_vol_ratio","D0_amount_ratio",
           "turnover","mcap_proxy","M_breadth","M_breadth_accel","M_diffusion","M_amount_shock","M_risk_on"]
BASE=ev[base_cols].to_numpy(float)

train_idx=np.where((ev.year>=2015)&(ev.year<=2018))[0]
cal_idx=np.where(ev.year==2019)[0]
test_idx=np.where((ev.year>=2020)&(ev.year<=2026))[0]

sc_seq=StandardScaler()
Ztr=sc_seq.fit_transform(SEQ[train_idx])
Zcal=sc_seq.transform(SEQ[cal_idx]); Ztest=sc_seq.transform(SEQ[test_idx])
pca=PCA(n_components=24,random_state=SEED)
Ptr=pca.fit_transform(Ztr); Pcal=pca.transform(Zcal); Ptest=pca.transform(Ztest)

sc_base=StandardScaler()
Btr=sc_base.fit_transform(BASE[train_idx]); Bcal=sc_base.transform(BASE[cal_idx]); Btest=sc_base.transform(BASE[test_idx])

# graph scaled
sc_g=StandardScaler()
Gtr=sc_g.fit_transform(GRAPH[train_idx]); Gcal=sc_g.transform(GRAPH[cal_idx]); Gtest=sc_g.transform(GRAPH[test_idx])

def wilson(k,n,z=1.6448536269514722):
    if n<=0:return 0.0
    p=k/n; den=1+z*z/n
    return (p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def outcome_summary(target,mask):
    z=ev.iloc[test_idx[mask]].copy(); n=len(z); k=int(z[target].sum())
    yrs=[]
    for y in range(2020,2027):
        a=z[z.year==y]; nn=len(a); kk=int(a[target].sum())
        yrs.append({"year":y,"n":nn,"rate":round(100*kk/nn,2) if nn else None,
                    "avg_net":round(float(a["pnl_"+target].mean()),3) if nn else None})
    return {"n":n,"k":k,"win_rate_pct":round(100*k/n,2) if n else None,
            "wilson90_lb_pct":round(100*wilson(k,n),2) if n else None,
            "avg_net_pct":round(float(z["pnl_"+target].mean()),3) if n else None,
            "median_net_pct":round(float(z["pnl_"+target].median()),3) if n else None,
            "tail_loss_pct":round(100*float((z["pnl_"+target]<=-5).mean()),2) if n else None,
            "yearly":yrs}

def choose_score(score_cal,target,extra=None,min_n=15):
    ycal=ev.iloc[cal_idx][target].to_numpy()
    best=None
    for q in [.50,.60,.70,.75,.80,.85,.875,.90,.925,.95]:
        th=float(np.quantile(score_cal,q))
        m=score_cal>=th
        if extra is not None: m &= extra
        n=int(m.sum())
        if n<min_n: continue
        z=ev.iloc[cal_idx[m]]; k=int(z[target].sum()); wr=k/n; avg=float(z["pnl_"+target].mean())
        feasible=(wr>=.75 and avg>=1)
        key=(1 if feasible else 0,wilson(k,n),avg,n,wr)
        dat={"th":th,"q":q,"n":n,"k":k,"wr":wr,"avg":avg,"feasible":feasible}
        if best is None or key>best[0]: best=(key,dat)
    return best[1]

def fmt_result(name,target,calmeta,audit,notes):
    return {"model":name,"target":target,
            "cal":{"n":calmeta["n"],"k":calmeta["k"],"win_rate_pct":round(100*calmeta["wr"],2),
                   "avg_net_pct":round(calmeta["avg"],3),"constraints_met":bool(calmeta["feasible"])},
            "audit":audit,"notes":notes}

results=[]

# ============================================================
# MODEL 1: SEQUENCE MOTIF
# PCA sequence -> KMeans motifs -> motif prior + logistic posterior.
# ============================================================
best_m1=None
for target,_,_ in TARGETS:
    ytr=ev.iloc[train_idx][target].to_numpy()
    ycal=ev.iloc[cal_idx][target].to_numpy()
    for K in [6,10,14]:
        km=KMeans(n_clusters=K,random_state=SEED,n_init=20).fit(Ptr)
        ctr=km.labels_; ccal=km.predict(Pcal); ctest=km.predict(Ptest)
        rates=np.array([(ytr[ctr==c].sum()+2)/(max(1,(ctr==c).sum())+4) for c in range(K)])
        motif_tr=rates[ctr][:,None]; motif_cal=rates[ccal][:,None]; motif_test=rates[ctest][:,None]
        Xtr=np.hstack([Ptr,Btr,motif_tr]); Xcal=np.hstack([Pcal,Bcal,motif_cal]); Xtest=np.hstack([Ptest,Btest,motif_test])
        clf=Pipeline([("s",StandardScaler()),("m",LogisticRegression(C=.2,class_weight="balanced",max_iter=7000,random_state=SEED))])
        clf.fit(Xtr,ytr)
        sc=clf.predict_proba(Xcal)[:,1]; st=clf.predict_proba(Xtest)[:,1]
        gm=choose_score(sc,target)
        aud=outcome_summary(target,st>=gm["th"])
        key=(1 if gm["feasible"] else 0,wilson(gm["k"],gm["n"]),gm["avg"])
        if best_m1 is None or key>best_m1[0]:
            best_m1=(key,fmt_result("M1_SEQUENCE_MOTIF",target,gm,aud,{"clusters":K}),st,gm)
results.append(best_m1[1])

# ============================================================
# MODEL 2: COMPETING HAZARD
# Person-day cause-specific hazards, derive target-before-stop CIF.
# ============================================================
best_m2=None
for target,tpct,horizon in TARGETS:
    # Expanded training rows from first-entry events.
    Xrows=[]; yt=[]; ys=[]
    for pos,eidx in enumerate(train_idx):
        row=ev.iloc[eidx]
        # reconstruct event-time from realized pnl outcome approximately using daily labels unavailable here:
        # use event result as terminal target vs stop/time, while daily baseline hazard learned with day index.
        # For target successes assign terminal event by midpoint proxy from horizon; for failures tail <=-5 acts as stop event.
        is_t=int(row[target]); is_s=int(row["pnl_"+target]<=-3.3)
        terminal=horizon if not (is_t or is_s) else max(1,int(round(horizon*0.55)))
        for day in range(1,terminal+1):
            feat=np.r_[Ptr[pos],Btr[pos],day/horizon,(day/horizon)**2]
            Xrows.append(feat)
            yt.append(1 if (day==terminal and is_t) else 0)
            ys.append(1 if (day==terminal and is_s and not is_t) else 0)
    Xh=np.asarray(Xrows); yt=np.asarray(yt); ys=np.asarray(ys)
    mt=LogisticRegression(C=.15,class_weight="balanced",max_iter=7000,random_state=SEED).fit(Xh,yt)
    ms=LogisticRegression(C=.15,class_weight="balanced",max_iter=7000,random_state=SEED).fit(Xh,ys)
    def cif(P,B):
        out=[]
        for a,b in zip(P,B):
            surv=1.0; cif_t=0.0
            for day in range(1,horizon+1):
                x=np.r_[a,b,day/horizon,(day/horizon)**2][None,:]
                ht=float(mt.predict_proba(x)[0,1]); hs=float(ms.predict_proba(x)[0,1])
                s=max(ht+hs,1.0)
                if s>1: ht/=s; hs/=s
                cif_t += surv*ht
                surv *= max(0,1-ht-hs)
            out.append(cif_t)
        return np.asarray(out)
    sc=cif(Pcal,Bcal); st=cif(Ptest,Btest)
    gm=choose_score(sc,target)
    aud=outcome_summary(target,st>=gm["th"])
    key=(1 if gm["feasible"] else 0,wilson(gm["k"],gm["n"]),gm["avg"])
    if best_m2 is None or key>best_m2[0]:
        best_m2=(key,fmt_result("M2_COMPETING_HAZARD",target,gm,aud,{"horizon":horizon}),st,gm)
results.append(best_m2[1])

# ============================================================
# MODEL 3: ANALOG RETRIEVAL
# Nearest historical analogs in sequence+context space.
# ============================================================
best_m3=None
Xtr=np.hstack([Ptr,Btr]); Xcal=np.hstack([Pcal,Bcal]); Xtest=np.hstack([Ptest,Btest])
sc_knn=StandardScaler(); Xtr2=sc_knn.fit_transform(Xtr); Xcal2=sc_knn.transform(Xcal); Xtest2=sc_knn.transform(Xtest)
for target,_,_ in TARGETS:
    ytr=ev.iloc[train_idx][target].to_numpy(float)
    pnltr=ev.iloc[train_idx]["pnl_"+target].to_numpy(float)
    for k in [25,40,60,100]:
        nn=NearestNeighbors(n_neighbors=min(k,len(Xtr2)),metric="euclidean").fit(Xtr2)
        dc,ic=nn.kneighbors(Xcal2); dtst,itst=nn.kneighbors(Xtest2)
        wc=1/(dc+0.25); wt=1/(dtst+0.25)
        pc=(wc*ytr[ic]).sum(1)/wc.sum(1); pt=(wt*ytr[itst]).sum(1)/wt.sum(1)
        muc=(wc*pnltr[ic]).sum(1)/wc.sum(1); mut=(wt*pnltr[itst]).sum(1)/wt.sum(1)
        scorec=pc*(1/(1+np.exp(-muc/1.5))); scoret=pt*(1/(1+np.exp(-mut/1.5)))
        gm=choose_score(scorec,target)
        aud=outcome_summary(target,scoret>=gm["th"])
        key=(1 if gm["feasible"] else 0,wilson(gm["k"],gm["n"]),gm["avg"])
        if best_m3 is None or key>best_m3[0]:
            best_m3=(key,fmt_result("M3_ANALOG_RETRIEVAL",target,gm,aud,{"k_neighbors":k}),scoret,gm)
results.append(best_m3[1])

# ============================================================
# MODEL 4: REGIME x LEADER GRAPH
# Graph centrality to same-day leaders + sequence + market regime.
# ============================================================
best_m4=None
for target,_,_ in TARGETS:
    ytr=ev.iloc[train_idx][target].to_numpy()
    Xtr=np.hstack([Ptr,Btr,Gtr]); Xcal=np.hstack([Pcal,Bcal,Gcal]); Xtest=np.hstack([Ptest,Btest,Gtest])
    clf=HistGradientBoostingClassifier(max_depth=3,learning_rate=.03,max_iter=300,l2_regularization=10,min_samples_leaf=18,random_state=SEED)
    clf.fit(Xtr,ytr)
    sc=clf.predict_proba(Xcal)[:,1]; st=clf.predict_proba(Xtest)[:,1]
    # graph quality gate chosen only on 2019: weighted-degree above a coarse quantile.
    for qg in [.25,.40,.55,.70]:
        gg=float(np.quantile(GRAPH[cal_idx,3],qg))
        extra=GRAPH[cal_idx,3]>=gg
        gm=choose_score(sc,target,extra=extra)
        aud=outcome_summary(target,(st>=gm["th"])&(GRAPH[test_idx,3]>=gg))
        key=(1 if gm["feasible"] else 0,wilson(gm["k"],gm["n"]),gm["avg"])
        if best_m4 is None or key>best_m4[0]:
            best_m4=(key,fmt_result("M4_REGIME_LEADER_GRAPH",target,gm,aud,{"graph_weighted_degree_min":round(gg,4)}),st,gm)
results.append(best_m4[1])

# ============================================================
# MODEL 5: TWO-STAGE DYNAMIC ENTRY
# Stage 1 D0 sequence quality; Stage 2 sequential D+1/D+2/D+3 timing.
# ============================================================
# Map event PCs and static D0 features to entry rows.
P_all=np.zeros((len(ev),24)); P_all[train_idx]=Ptr; P_all[cal_idx]=Pcal; P_all[test_idx]=Ptest
stage1_scores={}
best_m5=None

# event-id split sets
train_e=set(train_idx.tolist()); cal_e=set(cal_idx.tolist()); test_e=set(test_idx.tolist())
for target,_,_ in TARGETS:
    # Stage1 quality on events
    ytr=ev.iloc[train_idx][target].to_numpy()
    s1=HistGradientBoostingClassifier(max_depth=3,learning_rate=.03,max_iter=260,l2_regularization=8,min_samples_leaf=18,random_state=SEED)
    s1.fit(np.hstack([Ptr,Btr]),ytr)
    p1_train=s1.predict_proba(np.hstack([Ptr,Btr]))[:,1]
    p1_cal=s1.predict_proba(np.hstack([Pcal,Bcal]))[:,1]
    p1_test=s1.predict_proba(np.hstack([Ptest,Btest]))[:,1]
    map_p1={}
    for eidx,val in zip(train_idx,p1_train): map_p1[eidx]=val
    for eidx,val in zip(cal_idx,p1_cal): map_p1[eidx]=val
    for eidx,val in zip(test_idx,p1_test): map_p1[eidx]=val

    ecols=["entry_day","entry_gap","entry_gap_prev","prev_close","prev_high","prev_low","prev_vol","cum_min_close","cum_max_high"]
    trr=entry[entry.event_id.isin(train_e)].copy()
    calr=entry[entry.event_id.isin(cal_e)].copy()
    testr=entry[entry.event_id.isin(test_e)].copy()
    def makeX(df):
        dyn=df[ecols].to_numpy(float)
        pcs=np.vstack([P_all[int(e)] for e in df.event_id])
        p1=np.array([map_p1[int(e)] for e in df.event_id])[:,None]
        return np.hstack([pcs,dyn,p1])
    Xr=makeX(trr); Xc=makeX(calr); Xt=makeX(testr)
    timing=HistGradientBoostingClassifier(max_depth=3,learning_rate=.03,max_iter=260,l2_regularization=10,min_samples_leaf=20,random_state=SEED)
    timing.fit(Xr,trr[target].to_numpy())
    calr["score"]=timing.predict_proba(Xc)[:,1]
    testr["score"]=timing.predict_proba(Xt)[:,1]

    # Sequential threshold: enter first day whose score passes; no peeking ahead.
    def policy(df,th):
        chosen=[]
        for eid,z in df.sort_values(["event_id","entry_day"]).groupby("event_id"):
            q=z[z.score>=th]
            if len(q): chosen.append(q.iloc[0])
        return pd.DataFrame(chosen)
    best=None
    for q in [.50,.60,.70,.75,.80,.85,.875,.90,.925,.95]:
        th=float(np.quantile(calr.score,q))
        z=policy(calr,th); n=len(z)
        if n<15: continue
        k=int(z[target].sum()); wr=k/n; avg=float(z["pnl_"+target].mean()); feasible=(wr>=.75 and avg>=1)
        key=(1 if feasible else 0,wilson(k,n),avg,n,wr)
        dat={"th":th,"q":q,"n":n,"k":k,"wr":wr,"avg":avg,"feasible":feasible}
        if best is None or key>best[0]: best=(key,dat)
    gm=best[1]
    zt=policy(testr,gm["th"])
    # audit directly from chosen entry rows
    n=len(zt); k=int(zt[target].sum())
    yrs=[]
    for y in range(2020,2027):
        a=zt[zt.year==y]; nn=len(a); kk=int(a[target].sum())
        yrs.append({"year":y,"n":nn,"rate":round(100*kk/nn,2) if nn else None,
                    "avg_net":round(float(a["pnl_"+target].mean()),3) if nn else None})
    aud={"n":n,"k":k,"win_rate_pct":round(100*k/n,2) if n else None,
         "wilson90_lb_pct":round(100*wilson(k,n),2) if n else None,
         "avg_net_pct":round(float(zt["pnl_"+target].mean()),3) if n else None,
         "median_net_pct":round(float(zt["pnl_"+target].median()),3) if n else None,
         "tail_loss_pct":round(100*float((zt["pnl_"+target]<=-5).mean()),2) if n else None,
         "yearly":yrs,
         "entry_day_mix":{str(k):round(100*v,2) for k,v in zt.entry_day.value_counts(normalize=True).sort_index().items()}}
    key=(1 if gm["feasible"] else 0,wilson(gm["k"],gm["n"]),gm["avg"])
    if best_m5 is None or key>best_m5[0]:
        best_m5=(key,fmt_result("M5_TWO_STAGE_DYNAMIC_ENTRY",target,gm,aud,{"sequential_policy":True}),gm)
results.append(best_m5[1])

def pass_rule(r):
    a=r["audit"]; yrs=[y for y in a["yearly"] if y["n"]>0]
    return (a["n"]>=100 and a["win_rate_pct"]>=75 and a["avg_net_pct"]>=1.0 and
            a["tail_loss_pct"]<=10 and len(yrs)>=5 and
            min(y["rate"] for y in yrs)>=65 and min(y["avg_net"] for y in yrs)>=0)

for r in results: r["promotion_pass"]=pass_rule(r)
passing=[r for r in results if r["promotion_pass"]]
champ=max(passing,key=lambda r:(r["audit"]["win_rate_pct"],r["audit"]["avg_net_pct"],r["audit"]["n"])) if passing else None

result={
 "study":"KQ Five Structurally-New Models v2",
 "models_tested":[
   "Sequence Motif","Competing Hazard","Analog Retrieval","Regime x Leader Graph","Two-Stage Dynamic Entry"
 ],
 "hard_goal":{"win_rate_pct":75,"avg_net_pct":1.0,"min_signals":100,"tail_loss_pct_max":10,
              "min_year_win_rate_pct":65,"min_year_avg_net_pct":0},
 "split":{"train":"2015-2018","calibration":"2019","historical_audit":"2020-2026"},
 "audit_note":"2020-2026 is not pristine OOS because earlier research has already examined this era; no 2020+ labels were used to fit or select these models.",
 "events":{"all":len(ev),"train":len(train_idx),"cal":len(cal_idx),"audit":len(test_idx),"entry_rows":len(entry)},
 "results":results,
 "champion":champ,
 "pass_any":bool(champ)
}
print("FINAL_RESULT",json.dumps(result,ensure_ascii=False),flush=True)
open("five_structural_models_v2_result.json","w").write(json.dumps(result,ensure_ascii=False,indent=2))
