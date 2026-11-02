"""
Template for a hypothesis agent. Run:
    /opt/miniconda3/bin/python3 -I example_hypothesis.py [path/to/daily_series.csv] [n_surrogates]

The CSV (optional) must have columns: date (UTC calendar day, YYYY-MM-DD), value.
Without a CSV a synthetic AR(1) placeholder series is used (expected result: NULL).

Shows:
  1. aligning an exogenous DAILY series to forecast issue times with no look-ahead (global feature, all bins)
  2. building a PER-BIN feature strictly from past catalog events
  3. evaluate() vs the ETAS/clustering baseline on y_all and y_main
  4. surrogate_test() (circular time shift; bin permutation for the per-bin feature) + year block bootstrap
"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")  # -I drops cwd: add explicitly
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import harness as H

csv = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else None
N_SURR = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 50   # use 200 for the real run

# ---- panel (cached; weekly issue times, 7-day horizon, target M>=6.0) ------------------------------
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)

# ---- 1) exogenous daily series -> panel windows, ONLY data before t_start --------------------------
if csv:
    d = pd.read_csv(csv, parse_dates=["date"])
    daily = pd.Series(d["value"].to_numpy(float), index=d["date"], name="myseries")
else:
    rng = np.random.default_rng(42)
    idx = pd.date_range("1970-01-01", "2026-10-17", freq="D")
    x = np.zeros(len(idx))
    for i in range(1, len(idx)):
        x[i] = 0.98 * x[i - 1] + rng.normal()
    daily = pd.Series(x, index=idx, name="ar1")
# mean of the 30 complete UTC days ending 1 day before t_start (lag_days=1 safety margin)
f_global = H.align_daily_series(daily, panel, window_days=30, agg="mean", lag_days=1, name="x_series_30d")
# anomaly vs. its own trailing 365-day mean: also strictly causal (uses only earlier rows)
f_global["x_series_anom"] = f_global["x_series_30d"] - f_global["x_series_30d"].rolling(52, min_periods=10).mean().shift(1)
# NOTE: never use .rolling(center=True), interpolation across t_start, or whole-period z-scores fitted on test data.

# ---- 2) per-bin feature from the catalog, strictly before t_start ----------------------------------
ev = H.get_events(min_mag=4.7)                       # all events (not declustered: mainshock flags are non-causal)
deep = ev[ev.depth > 70]
f_bin = H.past_event_count(deep, panel, window_days=365, nbr=0, name="x_deep_365d")
f_bin["x_deep_365d"] = np.log1p(f_bin["x_deep_365d"])

# combine: global features broadcast to every bin via merge on t_start
feats = f_bin.merge(f_global[["t_start", "x_series_anom"]], on="t_start", how="left")

# ---- 3) evaluate ------------------------------------------------------------------------------------
out = {}
for target in ("y_all", "y_main"):
    r = H.evaluate(feats, target=target, model="logit", panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])               # IG(full vs base), resampling calendar years
    print(H.summarize(r)); print("  year-block bootstrap:", bb)
    out[target] = {k: v for k, v in r.items() if k != "preds"} | {"bootstrap": bb}

# ---- 4) surrogates (primary significance) -----------------------------------------------------------
# PRIMARY: mode="phase_randomize" (Fourier phase-randomized surrogates; calibrated for slow / random-walk-like
# series, unlike circular_shift which is anti-conservative for them -- see HARNESS.md section 6).
# a) global series: surrogate of the RAW daily series, re-aggregated exactly like the real one, via a builder.
#    builder(None) must return the REAL features; builder(rng) a surrogate.
raw_vals = daily.to_numpy(float)

def builder(rng):
    vals = raw_vals if rng is None else H.ft_surrogate(raw_vals, rng)
    s = pd.Series(vals, index=daily.index, name=daily.name)
    g = H.align_daily_series(s, panel, window_days=30, agg="mean", lag_days=1, name="x_series_30d")
    g["x_series_anom"] = g["x_series_30d"] - g["x_series_30d"].rolling(52, min_periods=10).mean().shift(1)
    return g[["t_start", "x_series_anom"]]

s1 = H.surrogate_test(builder=builder, n=N_SURR, mode="phase_randomize", target="y_all", panel=panel, seed=1)
print("global series, phase_randomize (raw):", {k: v for k, v in s1.items() if k != "null"})
# b) per-bin feature: phase randomization of the panel feature (same phases in all bins) AND bin permutation
s2 = H.surrogate_test(f_bin, n=N_SURR, mode="phase_randomize", target="y_all", panel=panel, seed=2)
s3 = H.surrogate_test(f_bin, n=N_SURR, mode="bin_permute", target="y_all", panel=panel, seed=3)
print("per-bin, phase_randomize:", {k: v for k, v in s2.items() if k != "null"})
print("per-bin, bin permute    :", {k: v for k, v in s3.items() if k != "null"})
# repeat the surrogate tests with target="y_main" (mainshocks) -- that is where a genuine precursor should show.
print(f"Bonferroni alpha = {H.ALPHA_BONF:.4f}")
