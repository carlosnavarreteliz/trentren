"""Robustness: alternate split. Train t_start+h <= 2005-01-01, test 2005-01-01 <= t_start, t_start+h <= 2013-01-01.
(Never touches 2013+.) Caveat: p_poisson_* in the cached panel was estimated on 1976-2012 (includes this test window);
it enters base and full equally.
Usage: python -I run_altsplit.py <cat1> <cat2> <n_shift>
"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H
from tele_features import load_global, TeleFeatures

HERE = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering"
c1, c2, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
p0 = H.get_panel(7, 7, 6.0)
panel = p0.copy()
h = pd.Timedelta(days=7)
split = np.full(len(panel), "gap", dtype=object)
split[(panel.t_start + h <= pd.Timestamp("2005-01-01")).to_numpy()] = "train"
split[((panel.t_start >= pd.Timestamp("2005-01-01")) & (panel.t_start + h <= pd.Timestamp("2013-01-01"))).to_numpy()] = "test"
panel["split"] = split
tf = TeleFeatures(load_global([c1, c2]))
out = {}
for hyp in ("h1", "h2"):
    build = (lambda sd: tf.h1(panel, float(sd))) if hyp == "h1" else (lambda sd: tf.h2(panel, float(sd)))
    for target in ("y_all", "y_main"):
        r = H.evaluate(build(0), target=target, panel=panel, return_preds=True)
        bb = H.block_bootstrap_ci(r.pop("preds"))
        s = H.surrogate_test(builder=build, n=n, mode="circular_shift", target=target, panel=panel, seed=21, n_jobs=2)
        out[f"{hyp}_{target}"] = {"ig": r["ig_per_event"], "n_test_events": r["n_test_events"],
                                  "auc_base": r["base"]["roc_auc"], "auc_full": r["full"]["roc_auc"],
                                  "coef": r.get("coef_extra"), "boot": bb, "p_event_shift": s["p_value"],
                                  "null_mean": s["null_mean"], "null_q95": s["null_q95"]}
        print(hyp, target, out[f"{hyp}_{target}"], flush=True)
json.dump(out, open(f"{HERE}/results/altsplit_2005_2012.json", "w"), indent=1, default=float)
