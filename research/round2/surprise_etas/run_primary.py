"""Preregistered primary test: x_surprise, weekly M6 panel, logit, y_main (Bonferroni primary) and y_all.
Usage: python3 -I run_primary.py <features.parquet> <out.json> [n_phase]
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H

fpath, outp = sys.argv[1], sys.argv[2]
NPH = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
F = pd.read_parquet(fpath)
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
feat = F[["bin_id", "t_start", "x_surprise"]]
res = {}
for target in ("y_main", "y_all"):
    t0 = time.time()
    r = H.evaluate(feat, target=target, model="logit", panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    print(H.summarize(r), flush=True)
    sp = H.surrogate_test(feat, n=NPH, mode="phase_randomize", target=target, panel=panel, seed=11)
    bp = H.surrogate_test(feat, n=500, mode="bin_permute", target=target, panel=panel, seed=12)
    cs = H.surrogate_test(feat, n=200, mode="circular_shift", target=target, panel=panel, seed=13)
    g = H.evaluate(feat, target=target, model="gbm", panel=panel)
    strip = lambda d: {k: v for k, v in d.items() if k != "null"}
    res[target] = {
        "ig_per_event": r["ig_per_event"], "ig_full_vs_etas": r["ig_full_vs_etas"],
        "n_test_events": r["n_test_events"], "coef_extra": r.get("coef_extra"),
        "auc_base": r["base"]["roc_auc"], "auc_full": r["full"]["roc_auc"],
        "molchan_base": r["base"]["molchan_ass"], "molchan_full": r["full"]["molchan_ass"],
        "bootstrap": bb, "phase_randomize": strip(sp), "bin_permute": strip(bp), "circular_shift": strip(cs),
        "gbm_ig": g["ig_per_event"], "ttest_naive_do_not_use": r["ttest_full_vs_base"],
        "seconds": time.time() - t0,
    }
    # drop-one-test-year
    P = r["preds"]
    loo = {}
    for yr in sorted(P.year.unique()):
        Q = P[P.year != yr]
        n = Q.y.sum()
        ll = lambda p: np.sum(Q.y * np.log(Q[p]) + (1 - Q.y) * np.log(1 - Q[p]))
        loo[int(yr)] = float((ll("p_full") - ll("p_base")) / max(n, 1))
    res[target]["drop_one_year_ig"] = loo
    print(target, json.dumps({k: v for k, v in res[target].items() if k != "drop_one_year_ig"}, default=str), flush=True)
json.dump(res, open(outp, "w"), indent=1, default=str)
