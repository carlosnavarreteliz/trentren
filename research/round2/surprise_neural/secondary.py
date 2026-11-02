"""Secondary / robustness analyses. Usage: python3 -I secondary.py FEATS.parquet PART [N]
PART = singles | robust"""
import sys, os, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import harness as H

FP, PART = sys.argv[1], sys.argv[2]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
HERE = os.path.dirname(os.path.abspath(__file__))
feats = pd.read_parquet(FP).sort_values(["t_start", "bin_id"]).reset_index(drop=True)
P6 = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
PRIM = ["x_ntpp_logp60", "x_resid_90", "x_ae_score"]
EXPL = ["x_resid_30", "x_resid_365", "x_lam_90", "x_surp_mean_365", "x_surp_mag_365", "x_surp_time_365",
        "x_upd_norm_365"]


def slim(r):
    return {k: v for k, v in r.items() if k not in ("preds", "null")}


def detrend(df, cols):
    d = df[["bin_id", "t_start"] + cols].copy()
    for c in cols:
        tm = d.groupby("bin_id")[c].transform(lambda s: s.rolling(52, min_periods=10).mean().shift(1))
        d[c] = d[c] - tm
    return d


out = {}
if PART.startswith("singles"):
    for c in (PRIM if PART == "singles_prim" else EXPL):
        X = feats[["bin_id", "t_start", c]]
        for target in ("y_main", "y_all"):
            r = H.evaluate(X, target=target, panel=P6)
            s = H.surrogate_test(X, n=NS, mode="phase_randomize", target=target, panel=P6, seed=21, n_jobs=2)
            rec = {"ig": r["ig_per_event"], "coef": r["coef_extra"], "auc_base": r["base"]["roc_auc"],
                   "auc_full": r["full"]["roc_auc"], "p_phase": s["p_value"], "null_mean": s["null_mean"]}
            if c in PRIM:
                sb = H.surrogate_test(X, n=NS, mode="bin_permute", target=target, panel=P6, seed=22, n_jobs=2)
                rec["p_binperm"] = sb["p_value"]
            out[f"{c}|{target}"] = rec
            print(c, target, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, f"{PART}.json"), "w"), indent=1, default=float)

if PART == "robust":
    X = feats[["bin_id", "t_start"] + PRIM]
    # 1. GBM
    for target in ("y_main", "y_all"):
        r = H.evaluate(X, target=target, model="gbm", panel=P6)
        out[f"gbm|{target}"] = {"ig": r["ig_per_event"], "auc_base": r["base"]["roc_auc"], "auc_full": r["full"]["roc_auc"],
                                "importance": r["importance_extra"]}
        print("gbm", target, out[f"gbm|{target}"], flush=True)
    # 2. M5.5 panel (NTPP probability at M>=5.5)
    P55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
    X55 = feats[["bin_id", "t_start", "x_ntpp_logp55", "x_resid_90", "x_ae_score"]]
    for target in ("y_main", "y_all"):
        r = H.evaluate(X55, target=target, panel=P55, return_preds=True)
        s = H.surrogate_test(X55, n=NS, mode="phase_randomize", target=target, panel=P55, seed=31, n_jobs=2)
        out[f"M5.5|{target}"] = {"ig": r["ig_per_event"], "n": r["n_test_events"], "p_phase": s["p_value"],
                                 "auc_base": r["base"]["roc_auc"], "auc_full": r["full"]["roc_auc"],
                                 "boot": H.block_bootstrap_ci(r["preds"]), "coef": r["coef_extra"]}
        print("M5.5", target, out[f"M5.5|{target}"], flush=True)
    # 3. causally detrended block
    Xd = detrend(feats, PRIM)
    for target in ("y_main", "y_all"):
        r = H.evaluate(Xd, target=target, panel=P6, return_preds=True)
        s = H.surrogate_test(Xd, n=NS, mode="phase_randomize", target=target, panel=P6, seed=41, n_jobs=2)
        out[f"detrended|{target}"] = {"ig": r["ig_per_event"], "p_phase": s["p_value"], "coef": r["coef_extra"],
                                      "boot": H.block_bootstrap_ci(r["preds"])}
        print("detrended", target, out[f"detrended|{target}"], flush=True)
    # 4. circular shift (end-matched) and bin permute for the primary block
    for target in ("y_main", "y_all"):
        s = H.surrogate_test(X, n=NS, mode="circular_shift", target=target, panel=P6, seed=51, n_jobs=2)
        sb = H.surrogate_test(X, n=NS, mode="bin_permute", target=target, panel=P6, seed=52, n_jobs=2)
        out[f"circ_binperm|{target}"] = {"ig": s["ig_obs"], "p_circular": s["p_value"], "p_binperm": sb["p_value"]}
        print("circ/binperm", target, out[f"circ_binperm|{target}"], flush=True)
    # 5. drop-one-test-year and exclusion of post-M6 cells (from the primary fits)
    for target in ("y_main", "y_all"):
        pr = pd.read_parquet(os.path.join(HERE, f"preds_primary_{target}.parquet"))
        d = H._ll(pr.y.to_numpy(), pr.p_full.to_numpy()) - H._ll(pr.y.to_numpy(), pr.p_base.to_numpy())
        g = pd.DataFrame({"year": pr.year, "d": d, "y": pr.y}).groupby("year").sum()
        loo = {int(yv): float((g.d.sum() - g.d[yv]) / (g.y.sum() - g.y[yv])) for yv in g.index}
        te = P6[P6.split == "test"]
        recent = (te.log_days_since_m6_nbr.to_numpy() < np.log10(1 + 30))
        keep = ~recent
        out[f"loo_year|{target}"] = loo
        out[f"no_recent_m6|{target}"] = {"ig": float(d[keep].sum() / pr.y.to_numpy()[keep].sum()),
                                          "n_events": int(pr.y.to_numpy()[keep].sum()),
                                          "ig_recent_only": float(d[recent].sum() / max(pr.y.to_numpy()[recent].sum(), 1)),
                                          "n_recent_events": int(pr.y.to_numpy()[recent].sum())}
        print("loo", target, {k: round(v, 4) for k, v in loo.items()}, flush=True)
        print("no_recent_m6", target, out[f"no_recent_m6|{target}"], flush=True)
    # 6. feature level shift train vs test (descriptive)
    m = P6[["bin_id", "t_start", "split"]].merge(feats, on=["bin_id", "t_start"])
    lev = {}
    for c in PRIM + EXPL:
        a, b = m.loc[m.split == "train", c], m.loc[m.split == "test", c]
        lev[c] = {"train_mean": float(a.mean()), "test_mean": float(b.mean()), "train_sd": float(a.std()),
                  "shift_in_sd": float((b.mean() - a.mean()) / a.std())}
    out["level_shift"] = lev
    print("level shift", json.dumps(lev, indent=0), flush=True)
    json.dump(out, open(os.path.join(HERE, "robust.json"), "w"), indent=1, default=float)
