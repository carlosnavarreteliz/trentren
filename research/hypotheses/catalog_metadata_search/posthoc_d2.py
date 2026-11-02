"""POST-HOC (not preregistered as primary): causally detrended epicentral correlation dimension x_n1_sp_d2_1095.
usage: python -I posthoc_d2.py FEATURES_PARQUET OUT_JSON"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/catalog_metadata_search")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H, cmlib as C
F = pd.read_parquet(sys.argv[1]); c = "x_n1_sp_d2_1095"; out = {}
for pk, (s, h, m) in {"w6": (7, 7, 6.0), "w55": (7, 7, 5.5)}.items():
    P = H.get_panel(s, h, m)
    tr = (P.split == "train").to_numpy()
    X = C.winsorize(C.aligned_matrix(F, P, [c]), tr)
    G = pd.DataFrame(X[:, 0].reshape(-1, 29))
    det = (G - G.rolling(156, min_periods=52).mean().shift(1)).to_numpy().ravel()
    ef = C.frame(P, det[:, None], [c + "_det"])
    for tg in ("y_main", "y_all"):
        r = H.evaluate(ef, tg, "logit", P, return_preds=True)
        n = 1000 if (pk == "w6" and tg == "y_main") else 200
        sp = H.surrogate_test(ef, n=n, mode="phase_randomize", target=tg, panel=P, seed=41, n_jobs=2)
        sb = H.surrogate_test(ef, n=200, mode="bin_permute", target=tg, panel=P, seed=42, n_jobs=2)
        bb = H.block_bootstrap_ci(r["preds"])
        pr = r["preds"]; d = H._ll(pr.y.values, pr.p_full.values) - H._ll(pr.y.values, pr.p_base.values)
        g = pd.DataFrame({"year": pr.year, "d": d, "y": pr.y}).groupby("year").sum()
        dy = {int(k): float((g.d.sum() - g.d[k]) / max(g.y.sum() - g.y[k], 1)) for k in g.index}
        out[f"{pk}_{tg}"] = {"ig": r["ig_per_event"], "coef": r["coef_extra"], "p_phase": sp["p_value"], "n_phase": n,
                             "p_binperm": sb["p_value"], "boot": bb, "drop_year_min": min(dy.values()),
                             "auc_base": r["base"]["roc_auc"], "auc_full": r["full"]["roc_auc"]}
        print(pk, tg, json.dumps(out[f"{pk}_{tg}"]), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
