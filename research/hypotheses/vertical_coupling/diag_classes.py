import sys; sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/vertical_coupling")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H, vc_features as V
ev = V.classify(H.get_events(min_mag=4.7))
ev["dec"] = (ev.time.dt.year // 10) * 10
print(pd.crosstab(ev.dec, ev.cls))
print("slab_z NaN frac", np.isnan(ev.slab_z).mean())
print(ev.groupby("cls").depth.describe())
print(pd.crosstab(ev.bin_id, ev.cls).T)
g = V.load_gcmt(); g = V.classify(g.assign(mag=g.mw))
print(len(g)); print(pd.crosstab(g.cls, g.mech)); print(g.time.min(), g.time.max())
ev.to_parquet(sys.argv[1] if len(sys.argv) > 1 else "/tmp/x.parquet")
