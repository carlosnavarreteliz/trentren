"""PRIMARY test (preregistered): Marchitelli et al. (2020) solar-wind proton density, 1-day lead.

usage: python -I run_primary.py <space_weather_daily.csv> <out_json> <target> <n_surr> [mag=6.0] [mode=ft|shift]
Feature x_np_lag1 = log10 daily mean Np on the UTC day [t_start-1d, t_start) on the daily panel (step=1, h=1).
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/space_weather")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import harness as H
import sw_lib as L

csv, out_json, target, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
mag = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
mode = sys.argv[6] if len(sys.argv) > 6 else "ft"

daily = L.load_daily(csv)
panel = H.get_panel(step_days=1, horizon_days=1, target_mag=mag)
SPEC = ("lognp", None, dict(window_days=1, agg="mean", lag_days=0))
feat = L.make_feature(daily, "x_np_lag1", panel, spec=SPEC)
print("NaN fraction train/test:",
      feat[feat.t_start < H.TEST_START]["x_np_lag1"].isna().mean().round(3),
      feat[feat.t_start >= H.TEST_START]["x_np_lag1"].isna().mean().round(3), flush=True)
t0 = time.time()
res = {}
r = H.evaluate(feat, target=target, model="logit", panel=panel, return_preds=True)
bb = H.block_bootstrap_ci(r["preds"])
print(H.summarize(r), "\n bootstrap", bb, f"({time.time()-t0:.1f}s)", flush=True)
res["evaluate"] = {k: v for k, v in r.items() if k != "preds"} | {"bootstrap": bb}
if n > 0:
    b = L.builder_for(daily, "x_np_lag1", panel, spec=SPEC, kind=mode)
    sm = "phase_randomize" if mode == "ft" else "circular_shift"
    s = H.surrogate_test(builder=b, n=n, mode=sm, target=target, panel=panel, seed=11, n_jobs=1)
    print("surrogate", {k: v for k, v in s.items() if k != "null"}, flush=True)
    res["surrogate"] = s
json.dump(res, open(out_json, "w"), indent=1, default=float)
print("done", time.time() - t0)
