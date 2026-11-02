"""Secondary / robustness analyses (exploratory, no multiplicity credit).
Usage: python3 -I run_secondary.py <data_dir> <part> <out.json> [n_surr]
  part = components | variants
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/surprise_etas")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H

dd, part, outp = sys.argv[1], sys.argv[2], sys.argv[3]
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 200
F = pd.read_parquet(f"{dd}/features_m4.7_s7_h7.parquet")
pw = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)


def run(feat, panel, target, label, n=NS, surr=True, gbm=False):
    t0 = time.time()
    r = H.evaluate(feat, target=target, model="logit", panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    o = {"label": label, "target": target, "ig": r["ig_per_event"], "n_test_events": r["n_test_events"],
         "auc_base": r["base"]["roc_auc"], "auc_full": r["full"]["roc_auc"], "ci": [bb.get("ci_lo"), bb.get("ci_hi")],
         "bootstrap": bb, "coef": r.get("coef_extra")}
    if surr:
        s = H.surrogate_test(feat, n=n, mode="phase_randomize", target=target, panel=panel, seed=21)
        o["p_phase"] = s["p_value"]; o["null_mean"] = s["null_mean"]
    if gbm:
        o["gbm_ig"] = H.evaluate(feat, target=target, model="gbm", panel=panel)["ig_per_event"]
    o["sec"] = round(time.time() - t0, 1)
    print(json.dumps(o, default=str), flush=True)
    return o


out = []
if part == "components":
    comps = ["x_rr30", "x_rr90", "x_rr365", "x_lld30", "x_lld90", "x_lld365", "x_ad", "x_gap", "x_mag", "x_spat",
             "x_depth"]
    for tg in ("y_main", "y_all"):
        for c in comps:
            out.append(run(F[["bin_id", "t_start", c]], pw, tg, c))
        out.append(run(F[["bin_id", "t_start"] + comps], pw, tg, "block_all_components", gbm=True))
elif part == "variants":
    # causally detrended composite (trailing 260-week mean in the same bin, past only)
    D = F[["bin_id", "t_start", "x_surprise"]].sort_values(["bin_id", "t_start"]).copy()
    D["x_surprise_dt"] = D["x_surprise"] - D.groupby("bin_id")["x_surprise"].transform(
        lambda s: s.rolling(260, min_periods=52).mean().shift(1))
    F5 = pd.read_parquet(f"{dd}/features_m5.0_s7_h7.parquet")
    p55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
    Fd = pd.read_parquet(f"{dd}/features_m4.7_s1_h1.parquet")
    pd1 = H.get_panel(step_days=1, horizon_days=1, target_mag=6.0)
    for tg in ("y_main", "y_all"):
        out.append(run(D[["bin_id", "t_start", "x_surprise_dt"]], pw, tg, "composite_detrended_260w"))
        out.append(run(F5[["bin_id", "t_start", "x_surprise"]], pw, tg, "composite_m0=5.0"))
        out.append(run(F[["bin_id", "t_start", "x_surprise"]], p55, tg, "composite_panel_M5.5"))
        out.append(run(Fd[["bin_id", "t_start", "x_surprise"]], pd1, tg, "composite_daily_M6", n=100))
json.dump(out, open(outp, "w"), indent=1, default=str)
