"""Primary test (see PREREGISTRATION.md). Usage: python -I run_primary.py <out.json> [n_big]"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/quiescence_migration")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import qm_features as Q

out_path = sys.argv[1]
N_BIG = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
F = Q.build(panel)
res = {}
for target in ("y_main", "y_all"):
    t0 = time.time()
    r = H.evaluate(F, target=target, model="logit", panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    print(H.summarize(r), "\n  bootstrap:", bb, flush=True)
    row = {k: v for k, v in r.items() if k != "preds"} | {"bootstrap": bb}
    n = N_BIG if r["ig_per_event"] > 0 else 200
    s_pr = H.surrogate_test(F, n=n, mode="phase_randomize", target=target, panel=panel, seed=11, n_jobs=2)
    print(f"  phase_randomize n={n}:", {k: v for k, v in s_pr.items() if k != "null"}, flush=True)
    nb = N_BIG if s_pr["p_value"] < 0.05 else 200
    s_bp = H.surrogate_test(F, n=nb, mode="bin_permute", target=target, panel=panel, seed=12, n_jobs=2)
    print(f"  bin_permute n={nb}:", {k: v for k, v in s_bp.items() if k != "null"}, flush=True)
    s_cs = H.surrogate_test(F, n=200, mode="circular_shift", target=target, panel=panel, seed=13, n_jobs=2)
    print("  circular_shift(end-matched) n=200:", {k: v for k, v in s_cs.items() if k != "null"}, flush=True)
    row["sur_phase"] = {k: v for k, v in s_pr.items() if k != "null"}
    row["sur_binperm"] = {k: v for k, v in s_bp.items() if k != "null"}
    row["sur_circ"] = {k: v for k, v in s_cs.items() if k != "null"}
    res[target] = row
    print(f"  [{target} done in {time.time()-t0:.0f}s]", flush=True)
    json.dump(res, open(out_path, "w"), indent=1, default=float)
