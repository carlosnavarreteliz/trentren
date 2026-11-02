"""Fit pooled (global) and Chile-only models, evaluate on Chile 2013+ vs the harness ETAS-logit baseline,
plus leave-one-margin-out (LOMO) evaluation.

Usage: python3 -I run_models.py <panels_dir> <out_dir> <target_mag> [--lomo] [--targets y_all,y_main]
"""
import sys, os, json, time, glob
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
from common import signflip_p, ll, level_match

PDIR, OUT, TM = sys.argv[1], sys.argv[2], float(sys.argv[3])
DO_LOMO = "--lomo" in sys.argv
TARGETS = ["y_all", "y_main"]
for a in sys.argv:
    if a.startswith("--targets="):
        TARGETS = a.split("=", 1)[1].split(",")
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
LOMO_SEEDS = [0]
GBM = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=300, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, n_jobs=2, verbose=-1, deterministic=True, force_row_wise=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ data
hp = H.get_panel(7, 7, TM)
BASE_COLS = H.baseline_feature_cols(hp)                      # 21 harness baseline columns
ECOLS = [c for c in hp.columns if c.startswith("etas_")]
P = {}
for f in sorted(glob.glob(os.path.join(PDIR, f"panel_*_m{TM}.parquet"))):
    code = os.path.basename(f).split("_")[1]
    d = pd.read_parquet(f)
    d["margin"] = code
    P[code] = d
D = pd.concat(P.values(), ignore_index=True)
XX = [c for c in D.columns if c.startswith("xx_")]
log("rows", len(D), "margins", sorted(P), "extras", XX)
CH = (D.margin == "CHL").to_numpy()
TR = (D.split == "train").to_numpy()
TE = (D.split == "test").to_numpy()


def transform(df, cols):
    X = H._transform_base(df, cols)
    for j, c in enumerate(cols):
        if c.startswith("xx_cnt"):
            X[:, j] = np.log1p(df[c].to_numpy(float))
    return X


XBASE = transform(D, BASE_COLS)
XEXT = transform(D, XX)
E = D[ECOLS].to_numpy()


def etas_lograte(target, train_mask):
    tgt = "all" if target == "y_all" else "main"
    r_bg = -np.log1p(-D[f"p_poisson_{tgt}"].to_numpy())
    w = H.fit_etas_additive(E[train_mask], r_bg[train_mask], D[target].to_numpy()[train_mask])
    rate = np.column_stack([r_bg, E]) @ w
    return np.log10(np.maximum(rate, 1e-8))


def fit_gbm(X, y, train_mask, pred_mask, seeds):
    import lightgbm as lgb
    p = np.zeros(pred_mask.sum())
    imp = np.zeros(X.shape[1])
    for s in seeds:
        clf = lgb.LGBMClassifier(random_state=s, **GBM)
        clf.fit(X[train_mask], y[train_mask])
        p += clf.predict_proba(X[pred_mask])[:, 1] / len(seeds)
        imp += clf.booster_.feature_importance("gain") / len(seeds)
    return p, imp


def fit_logit(X, y, train_mask, pred_mask):
    p, _, w = H._fit_predict(X[train_mask], y[train_mask], X[pred_mask], "logit")
    return p


def metrics(y, p):
    from sklearn.metrics import roc_auc_score
    return {"ll": float(ll(y, p).sum()), "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "molchan_ass": H.molchan_ass(y, p), "sum_p": float(p.sum()), "obs": int(y.sum())}


def compare(y, year, pa, pb):
    p, ig, peryear = signflip_p(y, pa, pb, year)
    bb = H.block_bootstrap_ci(pd.DataFrame({"y": y, "year": year, "a": pa, "b": pb}), a="a", b="b")
    # drop-one-year range
    D_, yrs = [], sorted(set(year))
    d = ll(y, pa) - ll(y, pb)
    drop = {}
    for yy in yrs:
        m = year != yy
        drop[int(yy)] = float(d[m].sum() / max(y[m].sum(), 1))
    lm_a, lm_b = level_match(y, pa), level_match(y, pb)
    ig_lm = float((ll(y, lm_a).sum() - ll(y, lm_b).sum()) / max(y.sum(), 1))
    return {"ig_per_event": ig, "p_signflip": p, "boot_ci": [bb["ci_lo"], bb["ci_hi"]], "p_boot_le0": bb["p_boot_le0"],
            "ig_level_matched_oracle": ig_lm, "drop_one_year_min": min(drop.values()),
            "drop_one_year_max": max(drop.values()), "drop_one_year": drop, "per_year_ll_diff": peryear}


results = {"target_mag": TM, "n_rows": int(len(D)), "margins": sorted(P), "base_cols": BASE_COLS, "extra_cols": XX,
           "gbm_params": GBM, "seeds": SEEDS}
all_preds = []
for target in TARGETS:
    y = D[target].to_numpy()
    R = {}
    # ---------------- reference: harness baseline on Chile test
    hres = H.evaluate(None, target=target, model="logit", panel=hp, return_preds=True)
    hpred = hres["preds"]
    chte = D[CH & TE][["bin_id", "t_start"]].reset_index(drop=True)
    hpred["t_start"] = pd.to_datetime(hpred["t_start"])
    m = chte.merge(hpred, on=["bin_id", "t_start"], how="left")
    assert m.p_base.notna().all() and (m.y.to_numpy() == y[CH & TE]).all(), "Chile test cells do not match harness"
    yc, yr = m.y.to_numpy(), m.year.to_numpy()
    preds = {"p_base": m.p_base.to_numpy(), "p_etas": m.p_etas.to_numpy(), "p_poisson": m.p_poisson.to_numpy()}
    log(target, "harness base", round(hres["ig_base_vs_poisson"], 4), "N", hres["n_test_events"])

    # ---------------- training masks
    strict = TR                                    # all margins, pre-2013 (purged) rows
    chile_tr = TR & CH
    nonstrict = (TR & CH) | (~CH & (TR | TE | (D.split == "gap").to_numpy()))
    pred_mask = CH & TE

    lr_pooled = etas_lograte(target, strict)
    lr_chile = etas_lograte(target, chile_tr)
    lr_ns = etas_lograte(target, nonstrict)
    Xp_ext = np.column_stack([XBASE, lr_pooled, XEXT])
    Xc_ext = np.column_stack([XBASE, lr_chile, XEXT])
    Xn_ext = np.column_stack([XBASE, lr_ns, XEXT])
    Xp_base = np.column_stack([XBASE, lr_pooled])
    Xc_base = np.column_stack([XBASE, lr_chile])
    names_ext = BASE_COLS + ["log_etas_rate"] + XX

    # M1 primary pooled GBM (strict); predict ALL test rows (for per-margin pooled evaluation)
    p_all_te, imp1 = fit_gbm(Xp_ext, y, strict, TE, SEEDS)
    te_idx = np.where(TE)[0]
    preds["p_m1_pooled_gbm"] = p_all_te[np.isin(te_idx, np.where(pred_mask)[0])]
    log(target, "M1 done")
    if not DO_LOMO:
        preds["p_m2_chile_gbm"], imp2 = fit_gbm(Xc_ext, y, chile_tr, pred_mask, SEEDS)
        log(target, "M2 done")
        preds["p_m3_pooled_logit"] = fit_logit(Xp_base, y, strict, pred_mask)
        preds["p_m3b_chile_logit_repl"] = fit_logit(Xc_base, y, chile_tr, pred_mask)   # should equal harness p_base
        preds["p_m4_pooled_gbm_nonstrict"], _ = fit_gbm(Xn_ext, y, nonstrict, pred_mask, SEEDS)
        log(target, "M4 done")
        preds["p_m5_pooled_gbm_basefeat"], _ = fit_gbm(Xp_base, y, strict, pred_mask, SEEDS)
        preds["p_m6_chile_gbm_basefeat"], _ = fit_gbm(Xc_base, y, chile_tr, pred_mask, SEEDS)
        log(target, "M5/M6 done")
        # calibration-in-the-large on Chile TRAIN rows (in-sample) for M1 and M2
        p1_tr, _ = fit_gbm(Xp_ext, y, strict, chile_tr, [0])
        p2_tr, _ = fit_gbm(Xc_ext, y, chile_tr, chile_tr, [0])
        R["chile_train_calibration"] = {"obs": int(y[chile_tr].sum()), "sum_p_m1": float(p1_tr.sum()),
                                        "sum_p_m2": float(p2_tr.sum())}

        R["metrics"] = {k: metrics(yc, v) for k, v in preds.items()}
        R["n_test_events"] = int(yc.sum())
        R["max_abs_diff_m3b_vs_harness_base"] = float(np.max(np.abs(preds["p_m3b_chile_logit_repl"] - preds["p_base"])))
        comps = {"PRIMARY_m1_vs_base": ("p_m1_pooled_gbm", "p_base"),
                 "m1_vs_m2_chile_gbm": ("p_m1_pooled_gbm", "p_m2_chile_gbm"),
                 "m2_vs_base": ("p_m2_chile_gbm", "p_base"),
                 "m3_pooled_logit_vs_base": ("p_m3_pooled_logit", "p_base"),
                 "m4_nonstrict_vs_base": ("p_m4_pooled_gbm_nonstrict", "p_base"),
                 "m5_pooled_basefeat_vs_base": ("p_m5_pooled_gbm_basefeat", "p_base"),
                 "m5_vs_m6_basefeat_pooled_vs_chile": ("p_m5_pooled_gbm_basefeat", "p_m6_chile_gbm_basefeat"),
                 "m1_vs_etas": ("p_m1_pooled_gbm", "p_etas"),
                 "base_vs_poisson": ("p_base", "p_poisson")}
        R["comparisons"] = {k: compare(yc, yr, preds[a], preds[b]) for k, (a, b) in comps.items()}
        R["importance_gain_m1"] = dict(sorted(zip(names_ext, np.round(imp1 / imp1.sum(), 4).tolist()),
                                              key=lambda t: -t[1]))
        R["importance_gain_m2"] = dict(sorted(zip(names_ext, np.round(imp2 / max(imp2.sum(), 1e-12), 4).tolist()),
                                              key=lambda t: -t[1]))
        for k in ["PRIMARY_m1_vs_base", "m1_vs_m2_chile_gbm", "m2_vs_base", "m3_pooled_logit_vs_base"]:
            c = R["comparisons"][k]
            log(target, k, "IG", round(c["ig_per_event"], 4), "p", c["p_signflip"], "CI", np.round(c["boot_ci"], 3))

    # ---------------- per-margin evaluation (pooled strict model on each margin's test rows) + LOMO
    if DO_LOMO:
        L = {}
        dte = D[TE].reset_index(drop=True)
        dte["p_pooled"] = p_all_te
        for code in sorted(P):
            mm = (D.margin == code).to_numpy()
            ym = y[mm & TE]
            if ym.sum() < 10:
                L[code] = {"skipped": f"only {int(ym.sum())} positive test cells"}
                continue
            yrm = D.year.to_numpy()[mm & TE]
            lr_loc = etas_lograte(target, TR & mm)
            Xl_base = np.column_stack([XBASE, lr_loc])
            Xl_ext = np.column_stack([XBASE, lr_loc, XEXT])
            p_loc_logit = fit_logit(Xl_base, y, TR & mm, mm & TE)
            p_pool = dte.loc[(dte.margin == code).to_numpy(), "p_pooled"].to_numpy()
            lr_lomo = etas_lograte(target, TR & ~mm)
            Xo_ext = np.column_stack([XBASE, lr_lomo, XEXT])
            p_lomo, _ = fit_gbm(Xo_ext, y, TR & ~mm, mm & TE, LOMO_SEEDS)
            p_loc_gbm, _ = fit_gbm(Xl_ext, y, TR & mm, mm & TE, LOMO_SEEDS)
            L[code] = {"n_test_events": int(ym.sum()), "n_train_events": int(y[TR & mm].sum()),
                       "pooled_vs_local_logit": compare(ym, yrm, p_pool, p_loc_logit),
                       "lomo_vs_local_logit": compare(ym, yrm, p_lomo, p_loc_logit),
                       "local_gbm_vs_local_logit": compare(ym, yrm, p_loc_gbm, p_loc_logit),
                       "pooled_vs_local_gbm": compare(ym, yrm, p_pool, p_loc_gbm),
                       "auc": {"local_logit": metrics(ym, p_loc_logit)["auc"], "pooled": metrics(ym, p_pool)["auc"],
                               "lomo": metrics(ym, p_lomo)["auc"], "local_gbm": metrics(ym, p_loc_gbm)["auc"]}}
            for kk in L[code]:
                if isinstance(L[code][kk], dict) and "per_year_ll_diff" in L[code][kk]:
                    L[code][kk].pop("drop_one_year")
            log(target, "LOMO", code, "N", int(ym.sum()),
                "pooled-local", round(L[code]["pooled_vs_local_logit"]["ig_per_event"], 4),
                "lomo-local", round(L[code]["lomo_vs_local_logit"]["ig_per_event"], 4),
                "localgbm-local", round(L[code]["local_gbm_vs_local_logit"]["ig_per_event"], 4))
        R["per_margin"] = L
    results[target] = R
    pr = pd.DataFrame({"bin_id": m.bin_id, "t_start": m.t_start, "year": yr, "y": yc, **preds})
    pr["target"] = target
    all_preds.append(pr)

tag = f"m{TM}" + ("_lomo" if DO_LOMO else "") + ("_" + "_".join(TARGETS) if len(TARGETS) == 1 else "")
json.dump(results, open(os.path.join(OUT, f"results_{tag}.json"), "w"), indent=1, default=float)
pd.concat(all_preds).to_parquet(os.path.join(OUT, f"chile_test_preds_{tag}.parquet"), index=False)
log("done")
