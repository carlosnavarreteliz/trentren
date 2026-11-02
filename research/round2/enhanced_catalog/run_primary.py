"""PRIMARY preregistered test + y_all. Usage: python -I run_primary.py <ipoc_catalog.txt> [n_surr_pos]"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research"); sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H, ec_load, ec_features as F
OUT = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog/"
NPOS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
cat = ec_load.load_ipoc(sys.argv[1])
P, ob = F.sub_panel(7, 7, 5.0, 22, 27)
f = F.build_features(cat, P, ob, 3.0, 365, "2008-01-01")
cols = ["x_act7", "x_act30", "x_dbpos"]
# restrict scoring to the catalog-covered test period: test rows end at 2021-12-24 by construction of sub_panel
res = {}
for tgt in ("y_main", "y_all"):
    t0 = time.time()
    r = H.evaluate(f, target=tgt, model="logit", panel=P, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    print(H.summarize(r)); print("  bootstrap", bb, f"{time.time()-t0:.1f}s")
    n = NPOS if r["ig_per_event"] > 0 else 200
    s = H.surrogate_test(f, n=n, mode="phase_randomize", target=tgt, panel=P, seed=11)
    sb = H.surrogate_test(f, n=200, mode="bin_permute", target=tgt, panel=P, seed=12)
    sc = H.surrogate_test(f, n=200, mode="circular_shift", target=tgt, panel=P, seed=13)
    print("  phase p", s["p_value"], "n", n, "| bin_permute p", sb["p_value"], "| circ p", sc["p_value"])
    pr = r.pop("preds")
    pr.to_csv(OUT + f"preds_primary_{tgt}.csv", index=False)
    res[tgt] = {k: v for k, v in r.items()} | {"bootstrap": bb,
               "sur_phase": {k: v for k, v in s.items() if k != "null"},
               "sur_binperm": {k: v for k, v in sb.items() if k != "null"},
               "sur_circ": {k: v for k, v in sc.items() if k != "null"}}
json.dump(res, open(OUT + "primary.json", "w"), indent=1, default=float)
