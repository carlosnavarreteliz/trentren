import sys; sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research"); sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog")
import numpy as np, pandas as pd, harness as H, ec_load
c = ec_load.load_ipoc(sys.argv[1])
print(c.shape, c.time.min(), c.time.max()); print(c.describe().T)
c["year"]=c.time.dt.year
g = c.groupby("year")
print(pd.DataFrame({"n":g.size(),"mc":g.mag.apply(lambda m: H.mc_maxc(m.values)),"n_ge3":g.mag.apply(lambda m:(m>=3).sum()),"n_ge2.5":g.mag.apply(lambda m:(m>=2.5).sum()),"maxml":g.mag.max()}))
# daily counts around Iquique and Tocopilla
for a,b in [("2014-03-01","2014-04-20"),("2007-11-05","2007-11-25")]:
    s=c[(c.time>=a)&(c.time<b)]; print(s.groupby(s.time.dt.date).agg(n=("mag","size"),n3=("mag",lambda m:(m>=3).sum()),mx=("mag","max")).to_string())
u = H.get_events(min_mag=4.7)
u = u[(u.latitude>-24.5)&(u.latitude<-18)&(u.time>="2007-01-01")&(u.time<"2022-01-01")]
print("USGS M>=4.7 in footprint", len(u), "M>=5", (u.mag>=5).sum(), "M>=6", (u.mag>=6).sum(), "main5", (u.mainshock&(u.mag>=5)).sum(), "main6", (u.mainshock&(u.mag>=6)).sum())
print(u[u.mag>=6][["time","latitude","longitude","depth","mag","mainshock"]].to_string())
# match USGS M>=5 to IPOC events within 60 s
ti = c.time.values.astype("datetime64[s]").astype(np.int64)
m=[]
for _,r in u[u.mag>=5].iterrows():
    t0=np.datetime64(r.time,"s").astype(np.int64); j=np.searchsorted(ti,t0-60); k=np.searchsorted(ti,t0+60)
    m.append((r.time, r.mag, c.mag.values[j:k].max() if k>j else np.nan))
m=pd.DataFrame(m,columns=["t","Mw","ML_ipoc"]); print("matched frac", m.ML_ipoc.notna().mean()); print(m.groupby(m.t.dt.year).ML_ipoc.apply(lambda s:s.notna().mean()))
print(m.dropna().assign(d=lambda x:x.ML_ipoc-x.Mw).d.describe())
