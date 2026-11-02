"""Build the causal feature library on the weekly grid and cache it.
usage: python -I build_features.py RAW_CSV OUT_PARQUET"""
import sys, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/catalog_metadata_search")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import features as FL
t0 = time.time()
F = FL.build_feature_library(sys.argv[1])
F.to_parquet(sys.argv[2], index=False)
x = [c for c in F.columns if c.startswith("x_")]
print(F.shape, len(x), "features", f"{time.time()-t0:.0f}s")
tr = F.t_start < "2013-01-01"
cov = pd.DataFrame({"nan_train": F.loc[tr, x].isna().mean(), "nan_all": F[x].isna().mean()})
print(cov.sort_values("nan_train").to_string())
