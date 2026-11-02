"""Sanity checks of the features (no target evaluation): distributions, NaN fractions, causality (truncation test)."""
import sys; sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research"); sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog")
import numpy as np, pandas as pd, harness as H, ec_load, ec_features as F
cat = ec_load.load_ipoc(sys.argv[1])
P, ob = F.sub_panel(7, 7, 5.0, 22, 27)
print(P.shape, P.t_start.min(), P.t_start.max(), P.split.value_counts().to_dict(), P.bin_id.unique())
f = F.build_features(cat, P, ob, 3.0, 365, "2008-01-01")
print(f.describe().T)
print(f[f.t_start>="2008-01-01"].isna().mean())
# causality: truncate catalog at a cut and recompute for t_start <= cut
cut = pd.Timestamp("2014-03-20")
f2 = F.build_features(cat[cat.time < cut], P, ob, 3.0, 365, "2008-01-01")
m = f.t_start <= cut
a, b = f[m].iloc[:, 2:].to_numpy(), f2[m].iloc[:, 2:].to_numpy()
print("max abs diff after truncation:", np.nanmax(np.abs(a-b)), "nan mismatch:", (np.isnan(a)!=np.isnan(b)).sum())
# feature values around Iquique for bin of -19.6 (orig 26 -> 4)
print(f[(f.bin_id==4)&(f.t_start>="2014-02-01")&(f.t_start<"2014-04-20")].to_string())
