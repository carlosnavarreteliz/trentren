"""Sanity + leakage checks (TRAIN-period descriptives only; no test targets inspected).
1. timing, NaN fractions, train-period summary stats and correlations with baseline features.
2. truncation test: re-decluster the catalog truncated at t*, rebuild features, compare rows with t_start <= t*.
"""
import sys, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/quiescence_migration")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import qm_features as Q

panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
t0 = time.time()
F = Q.build(panel)
print(f"built in {time.time()-t0:.1f}s, shape {F.shape}")
tr = F.t_start < H.TEST_START
print("NaN frac (all rows):", F[Q.ALL_COLS].isna().mean().round(3).to_dict())
print(F.loc[tr, Q.ALL_COLS].describe().T.round(3))
m = panel[["bin_id", "t_start"] + H.baseline_feature_cols(panel)].merge(F, on=["bin_id", "t_start"])
m = m[m.t_start < H.TEST_START]
base = ["cnt_bin_365d", "cnt_nbr_365d", "log_days_since_m6_nbr", "maxmag_nbr_365d", "log_bg_m45"]
for c in base:
    if c.startswith("cnt"):
        m[c] = np.log1p(m[c])
print("train corr with baseline features:\n", m[Q.ALL_COLS + base].corr().loc[Q.ALL_COLS, base].round(2))
print("train corr among features:\n", m[Q.ALL_COLS].corr().round(2))
F.to_parquet("/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/quiescence_migration/features_default.parquet")

# ---- truncation (look-ahead) test ----
cat = H.load_catalog(min_mag=4.7)
full_decl = H.decluster(method="zbz")
eta0 = full_decl.attrs["log10_eta0"]
ts = pd.DatetimeIndex(np.sort(panel.t_start.unique()))
for tstar in ["2010-02-25", "2014-03-20", "1995-07-28"]:
    tstar = pd.Timestamp(tstar)
    ct = cat[cat.time < tstar].reset_index(drop=True)
    d = H.decluster(cat=ct, method="zbz", eta0=eta0)
    bg = d[(d.bg) & (d.depth <= 70)].sort_values("time").reset_index(drop=True)
    big = d[(d.mag >= 7.0) & (d.depth <= 70)].sort_values("time").reset_index(drop=True)
    tsub = ts[ts <= tstar]
    Ft = Q.compute(bg, big, tsub)
    Ff = F[F.t_start <= tstar].reset_index(drop=True)
    diff = np.nanmax(np.abs(Ft[Q.ALL_COLS].to_numpy() - Ff[Q.ALL_COLS].to_numpy()))
    nanmis = (Ft[Q.ALL_COLS].isna().to_numpy() != Ff[Q.ALL_COLS].isna().to_numpy()).sum()
    print(f"truncation at {tstar.date()}: rows {len(Ft)}, max abs diff {diff:.2e}, NaN mismatches {nanmis}")
