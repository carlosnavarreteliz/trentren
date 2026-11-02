"""Primary preregistered test: weekly M>=6 panel, F1 = {x_ids_anom, x_or_anom, x_vert_shift}, logit.
Usage: python -I run_primary.py <out_dir> [n_phase] [n_other]
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/vertical_coupling")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import vc_features as V

out_dir = sys.argv[1]
N_PH = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
N_OT = int(sys.argv[3]) if len(sys.argv) > 3 else 500

panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
ev = V.classify(H.get_events(min_mag=4.7))
F1 = V.build_F1(panel, ev)
F1.to_parquet(f"{out_dir}/F1_weekly.parquet")
print(F1.describe().T, flush=True)

res = {}
for target in ("y_main", "y_all"):
    r = H.evaluate(F1, target=target, model="logit", panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    print(H.summarize(r), "\n boot:", bb, flush=True)
    r["preds"].to_parquet(f"{out_dir}/preds_primary_{target}.parquet")
    res[target] = {k: v for k, v in r.items() if k != "preds"} | {"bootstrap": bb}
    json.dump(res, open(f"{out_dir}/primary.json", "w"), indent=1, default=float)

for target in ("y_main", "y_all"):
    for mode, n in (("phase_randomize", N_PH), ("bin_permute", N_OT), ("circular_shift", N_OT)):
        t0 = time.time()
        s = H.surrogate_test(F1, n=n, mode=mode, target=target, panel=panel, seed=11, n_jobs=2)
        print(target, mode, {k: v for k, v in s.items() if k != "null"}, f"{time.time()-t0:.0f}s", flush=True)
        res[target][f"sur_{mode}"] = s
        json.dump(res, open(f"{out_dir}/primary.json", "w"), indent=1, default=float)
print("DONE", flush=True)
