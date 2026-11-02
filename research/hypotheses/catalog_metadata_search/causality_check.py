"""Leakage unit test: recompute features at sample issue times from a catalog TRUNCATED at t_start, compare to library.
usage: python -I causality_check.py RAW_CSV FEATURES_PARQUET"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/catalog_metadata_search")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H, features as FL
A, S = FL.load_events(sys.argv[1]); F = pd.read_parquet(sys.argv[2])
grid = pd.DatetimeIndex(np.sort(F.t_start.unique()))
rng = np.random.default_rng(5); worst = 0.0; n = 0
for k in rng.choice(np.arange(200, len(grid)), 6, replace=False):
    t = grid[k]; sub = grid[:k + 1]           # grid up to t (weekly-dynamics features need the history)
    At, St = A[A.time < t].reset_index(drop=True), S[S.time < t].reset_index(drop=True)
    for b in rng.choice(29, 3, replace=False):
        f = FL.neighbourhood_features(At, St, H.to_days(sub), int(b), 1, True)
        row = F[(F.t_start == t) & (F.bin_id == b)]
        for name, v in f.items():
            a, c = float(np.asarray(v)[-1]), float(row[f"x_n1_{name}"].iloc[0])
            if np.isnan(a) and np.isnan(c): continue
            d = abs(a - c) / (abs(c) + 1e-6); worst = max(worst, d); n += 1
            if d > 1e-4: print("MISMATCH", t, b, name, a, c)
print(f"compared {n} values, max rel diff {worst:.2e}")
