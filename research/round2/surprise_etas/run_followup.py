"""Follow-up checks (exploratory): (1) GBM surrogate test for the composite (GBM IG was +0.025 vs GBM baseline);
(2) x_ad (AD statistic of ETAS transformed times) which gave nominal p~0.02 on y_main among ~24 secondary tests.
Usage: python3 -I run_followup.py <data_dir> <out.json>
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H

dd, outp = sys.argv[1], sys.argv[2]
F = pd.read_parquet(f"{dd}/features_m4.7_s7_h7.parquet")
F5 = pd.read_parquet(f"{dd}/features_m5.0_s7_h7.parquet")
pw = H.get_panel()
strip = lambda d: {k: v for k, v in d.items() if k != "null"}
out = {}


def loo(P):
    o = {}
    for yr in sorted(P.year.unique()):
        Q = P[P.year != yr]
        ll = lambda p: np.sum(Q.y * np.log(Q[p]) + (1 - Q.y) * np.log(1 - Q[p]))
        o[int(yr)] = round(float((ll("p_full") - ll("p_base")) / max(Q.y.sum(), 1)), 4)
    return o


# (1) GBM surrogate for the composite
comp = F[["bin_id", "t_start", "x_surprise"]]
for tg in ("y_main", "y_all"):
    t0 = time.time()
    r = H.evaluate(comp, target=tg, model="gbm", panel=pw, return_preds=True)
    s = H.surrogate_test(comp, n=100, mode="phase_randomize", target=tg, model="gbm", panel=pw, seed=31)
    out[f"gbm_composite_{tg}"] = {"ig": r["ig_per_event"], "ig_base_vs_poisson": r["ig_base_vs_poisson"],
                                  "ig_full_vs_etas": r["ig_full_vs_etas"], "bootstrap": H.block_bootstrap_ci(r["preds"]),
                                  "surrogate": strip(s), "drop_one_year": loo(r["preds"]), "sec": time.time() - t0}
    print(tg, "gbm", json.dumps(out[f"gbm_composite_{tg}"], default=str), flush=True)

# (2) x_ad follow-up
D = F[["bin_id", "t_start", "x_ad"]].sort_values(["bin_id", "t_start"]).copy()
D["x_ad_dt"] = D["x_ad"] - D.groupby("bin_id")["x_ad"].transform(lambda s: s.rolling(260, min_periods=52).mean().shift(1))
variants = {"x_ad": F[["bin_id", "t_start", "x_ad"]], "x_ad_detrended": D[["bin_id", "t_start", "x_ad_dt"]],
            "x_ad_m0=5.0": F5[["bin_id", "t_start", "x_ad"]]}
for name, feat in variants.items():
    for tg in ("y_main", "y_all"):
        r = H.evaluate(feat, target=tg, model="logit", panel=pw, return_preds=True)
        n = 1000 if name == "x_ad" else 200
        s = H.surrogate_test(feat, n=n, mode="phase_randomize", target=tg, panel=pw, seed=41)
        o = {"ig": r["ig_per_event"], "bootstrap": H.block_bootstrap_ci(r["preds"]), "coef": r.get("coef_extra"),
             "p_phase": s["p_value"], "n": n, "drop_one_year": loo(r["preds"])}
        if name == "x_ad":
            o["p_bin_permute"] = H.surrogate_test(feat, n=300, mode="bin_permute", target=tg, panel=pw, seed=42)["p_value"]
        out[f"{name}_{tg}"] = o
        print(name, tg, json.dumps(o, default=str), flush=True)
p55 = H.get_panel(target_mag=5.5)
for tg in ("y_main", "y_all"):
    r = H.evaluate(variants["x_ad"], target=tg, model="logit", panel=p55)
    out[f"x_ad_M5.5_{tg}"] = {"ig": r["ig_per_event"]}
    print("x_ad M5.5", tg, r["ig_per_event"], flush=True)
json.dump(out, open(outp, "w"), indent=1, default=str)
