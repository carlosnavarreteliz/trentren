"""R3: alternate split. Selection was re-run inside 1976-2005 (select_features.py ... 2006); test once on 2006-01-01+.
usage: python -I final_alt.py FEATURES_PARQUET OUT_DIR TARGET [N_SURR]
"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/catalog_metadata_search")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import cmlib as C

feat_path, out_dir, target = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
H.TEST_START = pd.Timestamp("2006-01-01")
panel = H.build_panel(7, 7, 6.0, use_cache=False)
F = pd.read_parquet(feat_path)
sel = json.load(open(f"{out_dir}/selection_{target}_2006.json"))
tr = (panel.split == "train").to_numpy()
res = {"target": target, "split": "2006", "selected": sel["selected"], "top5_no_fdr": sel["top5_no_fdr"],
       "n_pass": sel["n_pass"], "min_p_boot": sel["min_p_boot"]}
rb = H.evaluate(None, target, "logit", panel)
res["baseline"] = {"n_test_events": rb["n_test_events"], "ig_base_vs_poisson": rb["ig_base_vs_poisson"]}
for name, cols in (("selected", sel["selected"]), ("top5_no_fdr", sel["top5_no_fdr"])):
    if not cols:
        res[name] = "empty -> null by construction"
        continue
    X = C.winsorize(C.aligned_matrix(F, panel, cols), tr)
    ef = C.frame(panel, X, cols)
    r = H.evaluate(ef, target, "logit", panel, return_preds=True)
    s = H.surrogate_test(ef, n=N, mode="phase_randomize", target=target, panel=panel, seed=31, n_jobs=2)
    res[name] = {"cols": cols, "ig_per_event": r["ig_per_event"], "n_test_events": r["n_test_events"],
                 "auc_base": r["base"]["roc_auc"], "auc_full": r["full"]["roc_auc"],
                 "boot": H.block_bootstrap_ci(r["preds"]), "p_phase": s["p_value"], "null_mean": s["null_mean"],
                 "coef": r["coef_extra"]}
    print(name, H.summarize(r, s), flush=True)
json.dump(res, open(f"{out_dir}/final_alt2006_{target}.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
