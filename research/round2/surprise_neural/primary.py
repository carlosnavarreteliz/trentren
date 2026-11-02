"""Preregistered primary test (y_main) and co-reported y_all, block {x_ntpp_logp60, x_resid_90, x_ae_score}.
Usage: python3 -I primary.py FEATS.parquet N_SURR"""
import sys, os, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import harness as H

FP = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
HERE = os.path.dirname(os.path.abspath(__file__))
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
feats = pd.read_parquet(FP)
COLS = ["x_ntpp_logp60", "x_resid_90", "x_ae_score"]
X = feats[["bin_id", "t_start"] + COLS]
out = {}
for target in ("y_main", "y_all"):
    r = H.evaluate(X, target=target, model="logit", panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    sur = H.surrogate_test(X, n=NS, mode="phase_randomize", target=target, panel=panel, seed=11, n_jobs=2)
    print(H.summarize(r, sur), flush=True)
    print("  bootstrap", bb, flush=True)
    r["preds"].to_parquet(os.path.join(HERE, f"preds_primary_{target}.parquet"), index=False)
    out[target] = {k: v for k, v in r.items() if k != "preds"} | {"bootstrap": bb} | \
                  {"surrogate": {k: v for k, v in sur.items() if k != "null"},
                   "null_quantiles": np.quantile(sur["null"], [0.05, 0.5, 0.95, 0.99]).tolist()}
json.dump(out, open(os.path.join(HERE, "primary.json"), "w"), indent=1, default=float)
