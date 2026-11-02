"""Primary test (preregistered). Usage:
  python -I run_primary.py <cat1.csv> <cat2.csv> <h1|h2> <y_all|y_main> <n_shift> <n_secondary> [step horizon mag dist_km]
Writes results/<hyp>_<target>_s<step>_m<mag>_d<dist>.json
"""
import sys, json, time, os
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H
from tele_features import load_global, TeleFeatures

HERE = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering"
c1, c2, hyp, target = sys.argv[1:5]
n_shift, n_sec = int(sys.argv[5]), int(sys.argv[6])
step, hor, mag = (int(sys.argv[7]), int(sys.argv[8]), float(sys.argv[9])) if len(sys.argv) > 9 else (7, 7, 6.0)
dist = float(sys.argv[10]) if len(sys.argv) > 10 else 1000.0
model = sys.argv[11] if len(sys.argv) > 11 else "logit"

panel = H.get_panel(step_days=step, horizon_days=hor, target_mag=mag)
tf = TeleFeatures(load_global([c1, c2]), dist_km=dist)


def build(shift_days=0.0):
    if hyp == "h1":
        return tf.h1(panel, shift_days)
    if hyp == "h2":
        return tf.h2(panel, shift_days)
    if hyp == "h2diff":
        return tf.h2(panel, shift_days, diff=True)
    raise ValueError(hyp)


feats = build(0.0)
t0 = time.time()
r = H.evaluate(feats, target=target, model=model, panel=panel, return_preds=True)
preds = r.pop("preds")
bb = H.block_bootstrap_ci(preds)
bb_etas = H.block_bootstrap_ci(preds, a="p_full", b="p_etas")
# leave-one-test-year-out IG (full vs base)
eps = 1e-12
def ll(y, p): return y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))
preds["d"] = ll(preds.y, preds.p_full) - ll(preds.y, preds.p_base)
loyo = {}
for yr in sorted(preds.year.unique()):
    m = preds.year != yr
    n = preds.y[m].sum()
    loyo[int(yr)] = float(preds.d[m].sum() / n) if n else None
per_year = preds.groupby("year").apply(lambda d: float(d.d.sum())).round(4).to_dict()
out = {"hyp": hyp, "target": target, "step": step, "horizon": hor, "mag": mag, "dist_km": dist, "model": model,
       "evaluate": r, "bootstrap_full_vs_base": bb, "bootstrap_full_vs_etas": bb_etas,
       "loyo_ig": loyo, "per_year_dll": {int(k): v for k, v in per_year.items()}}
print(H.summarize(r), "| boot", bb, flush=True)

if n_shift > 0:
    s = H.surrogate_test(builder=lambda sd: build(float(sd)), n=n_shift, mode="circular_shift", target=target,
                         model=model, panel=panel, seed=11, n_jobs=2)
    out["sur_event_shift"] = {k: v for k, v in s.items() if k != "null"}
    out["sur_event_shift_null"] = s["null"]
    print("event-time shift:", out["sur_event_shift"], flush=True)
if n_sec > 0:
    s = H.surrogate_test(feats, n=n_sec, mode="phase_randomize", target=target, model=model, panel=panel,
                         seed=12, n_jobs=2)
    out["sur_phase"] = {k: v for k, v in s.items() if k != "null"}
    print("phase_randomize:", out["sur_phase"], flush=True)
    if hyp == "h1":
        s = H.surrogate_test(feats, n=n_sec, mode="bin_permute", target=target, model=model, panel=panel,
                             seed=13, n_jobs=2)
        out["sur_bin_permute"] = {k: v for k, v in s.items() if k != "null"}
        print("bin_permute:", out["sur_bin_permute"], flush=True)
out["runtime_s"] = time.time() - t0
os.makedirs(f"{HERE}/results", exist_ok=True)
fn = f"{HERE}/results/{hyp}_{target}_s{step}_m{mag}_d{int(dist)}_{model}.json"
with open(fn, "w") as f:
    json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("wrote", fn)
