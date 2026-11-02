"""Harness evaluation of the neural forecasters.  Usage: python3 -I evaluate_nn.py WORK_DIR PART
PART = point | sur_gft_main | sur_gft_all | sur_others | robust
Writes WORK_DIR/../results/<PART>.json
"""
import sys, os, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H

WORK, PART = sys.argv[1], sys.argv[2]
RES = os.path.join(os.path.dirname(WORK.rstrip("/")), "results")
os.makedirs(RES, exist_ok=True)
t0 = time.time()
nn = pd.read_parquet(os.path.join(WORK, "nn_preds.parquet"))
nn["t_start"] = pd.to_datetime(nn["t_start"]).astype("datetime64[ns]")
P6 = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
HEAD = {("y_all", 6.0): "all6", ("y_main", 6.0): "main6", ("y_all", 5.5): "all55", ("y_main", 5.5): "main55"}
sig = lambda z: 1 / (1 + np.exp(-z))


def feat(kind, target, mag=6.0, demean=False):
    col = f"{kind}_{HEAD[(target, mag)]}"
    f = nn[["bin_id", "t_start", col]].rename(columns={col: f"x_nn_{kind}"}).copy()
    if demean:  # causal: minus own trailing 52-week mean in the same bin (rows strictly before)
        f = f.sort_values(["bin_id", "t_start"])
        f[f"x_nn_{kind}"] = f[f"x_nn_{kind}"] - f.groupby("bin_id")[f"x_nn_{kind}"].transform(
            lambda s: s.rolling(52, min_periods=10).mean().shift(1))
    return f


def clean(r):
    return {k: v for k, v in r.items() if k != "preds"}


def point():
    out = {}
    for target in ("y_all", "y_main"):
        for kind in ("gft", "co", "gzs"):
            r = H.evaluate(feat(kind, target), target=target, model="logit", panel=P6, return_preds=True)
            bb = H.block_bootstrap_ci(r["preds"])
            pr = r["preds"].merge(nn[["bin_id", "t_start", f"{kind}_{HEAD[(target, 6.0)]}"]], on=["bin_id", "t_start"])
            pr["p_nn"] = sig(pr[f"{kind}_{HEAD[(target, 6.0)]}"])
            ll = lambda p: float(H._ll(pr.y.to_numpy(), np.clip(p, 1e-9, 1 - 1e-9)).sum())
            N = int(pr.y.sum())
            from sklearn.metrics import roc_auc_score
            alone = dict(ig_nn_vs_base=(ll(pr.p_nn) - ll(pr.p_base)) / N, ig_nn_vs_etas=(ll(pr.p_nn) - ll(pr.p_etas)) / N,
                         ig_nn_vs_poisson=(ll(pr.p_nn) - ll(pr.p_poisson)) / N, auc_nn=float(roc_auc_score(pr.y, pr.p_nn)),
                         expected_nn=float(pr.p_nn.sum()), observed=N,
                         boot_nn_vs_base=H.block_bootstrap_ci(pr, a="p_nn", b="p_base"))
            # drop-one-test-year of the stacked IG
            d = H._ll(pr.y.to_numpy(), pr.p_full.to_numpy()) - H._ll(pr.y.to_numpy(), pr.p_base.to_numpy())
            g = pd.DataFrame({"year": pr.year, "d": d, "y": pr.y}).groupby("year").sum()
            drop = {int(y): float((g.d.sum() - g.d[y]) / max(1, g.y.sum() - g.y[y])) for y in g.index}
            per_year = {int(y): [round(float(g.d[y]), 3), int(g.y[y])] for y in g.index}
            # level shift of the feature
            f = feat(kind, target).merge(P6[["bin_id", "t_start", "split"]], on=["bin_id", "t_start"])
            lev = f.groupby("split")[f"x_nn_{kind}"].mean().to_dict()
            rd = H.evaluate(feat(kind, target, demean=True), target=target, model="logit", panel=P6)
            out[f"{kind}_{target}"] = dict(stacked=clean(r), bootstrap=bb, alone=alone, drop_one_year=drop,
                                           per_year_dLL_npos=per_year, feature_mean_by_split=lev,
                                           demeaned_ig=rd["ig_per_event"], demeaned_coef=rd.get("coef_extra"))
            print(target, kind, "IG stacked %.4f  CI [%.3f,%.3f]  alone-vs-base %.4f  AUC base %.3f full %.3f nn %.3f"
                  % (r["ig_per_event"], bb["ci_lo"], bb["ci_hi"], alone["ig_nn_vs_base"], r["base"]["roc_auc"],
                     r["full"]["roc_auc"], alone["auc_nn"]), flush=True)
            out[f"{kind}_{target}"]["_preds"] = pr[["bin_id", "t_start", "y", "p_base", "p_full", "p_nn"]]
        # direct contrast: does pooling help? gft vs co (stand-alone and stacked)
        a = out[f"gft_{target}"]["_preds"].rename(columns={"p_nn": "p_gft", "p_full": "p_full_gft"})
        b = out[f"co_{target}"]["_preds"][["bin_id", "t_start", "p_nn", "p_full"]].rename(
            columns={"p_nn": "p_co", "p_full": "p_full_co"})
        m = a.merge(b, on=["bin_id", "t_start"])
        m["year"] = pd.to_datetime(m.t_start).dt.year
        out[f"contrast_{target}"] = dict(alone_gft_vs_co=H.block_bootstrap_ci(m, a="p_gft", b="p_co"),
                                         stacked_gft_vs_co=H.block_bootstrap_ci(m, a="p_full_gft", b="p_full_co"))
        print(target, "contrast", out[f"contrast_{target}"], flush=True)
    # M>=7 descriptive (Chile test)
    from sklearn.metrics import roc_auc_score
    c = np.load(os.path.join(WORK, "chile_samples.npz"))
    te = c["split"] == "test"
    y7 = c["Y"][te, 4]
    rb = H.evaluate(None, target="y_all", model="logit", panel=P6, return_preds=True)["preds"]
    rate6 = -np.log1p(-rb.p_base.to_numpy())
    p7_gr = -np.expm1(-rate6 * 10 ** (-1.0))  # GR b=1 scaling of the baseline M6 probability
    m7 = {"n_pos_cells": int(y7.sum()), "pos_cells": [str(x)[:10] + f" bin{b}" for x, b in
                                                      zip(c["t_start"][te][y7 == 1], c["bin_id"][te][y7 == 1])]}
    for name, p in [("base_GR", p7_gr)] + [(k, sig(nn[f"{k}_all7"].to_numpy()[te])) for k in ("gft", "co", "gzs")]:
        m7[name] = dict(ll=float(H._ll(y7, np.clip(p, 1e-9, 1)).sum()), auc=float(roc_auc_score(y7, p)),
                        expected=float(p.sum()), p_at_pos=np.round(p[y7 == 1], 4).tolist())
    out["m7_descriptive"] = m7
    print("M7:", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'p_at_pos'})
                              for k, v in m7.items()}), flush=True)
    for k in list(out):
        if isinstance(out[k], dict):
            out[k].pop("_preds", None)
    return out


def sur(kind, target, n, mode="phase_randomize", panel=P6, mag=6.0, model="logit"):
    s = H.surrogate_test(feat(kind, target, mag), n=n, mode=mode, target=target, panel=panel, seed=11, model=model)
    s = {k: v for k, v in s.items() if k != "null"} | {"null_q99": float(np.quantile(s["null"], 0.99))}
    print(kind, target, mode, n, s, f"{time.time()-t0:.0f}s", flush=True)
    return s


if PART == "point":
    out = point()
elif PART == "sur_gft_main":
    out = {"gft_y_main_phase_n1000": sur("gft", "y_main", 1000)}
elif PART == "sur_gft_all":
    out = {"gft_y_all_phase_n1000": sur("gft", "y_all", 1000)}
elif PART == "sur_others":
    out = {f"{k}_{t}_phase_n200": sur(k, t, 200) for k in ("co", "gzs") for t in ("y_main", "y_all")}
elif PART == "robust":
    out = {}
    for t in ("y_main", "y_all"):
        out[f"gft_{t}_circshift_n200"] = sur("gft", t, 200, mode="circular_shift")
        r = H.evaluate(feat("gft", t), target=t, model="gbm", panel=P6)
        out[f"gft_{t}_gbm"] = clean(r)
        print(t, "gbm IG", r["ig_per_event"], flush=True)
    P55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
    for t in ("y_main", "y_all"):
        for k in ("gft", "co"):
            r = H.evaluate(feat(k, t, 5.5), target=t, model="logit", panel=P55, return_preds=True)
            out[f"{k}_{t}_m55"] = clean(r) | {"bootstrap": H.block_bootstrap_ci(r["preds"])}
            print(t, k, "M5.5 IG", r["ig_per_event"], flush=True)
        out[f"gft_{t}_m55_phase_n200"] = sur("gft", t, 200, panel=P55, mag=5.5)
json.dump(out, open(os.path.join(RES, f"{PART}.json"), "w"), indent=1, default=str)
print("done", PART, f"{time.time()-t0:.0f}s")
