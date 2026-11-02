"""Shared helpers for the audit of the earlier PI-ST-Fusion model (read-only use of src/)."""
import os
import sys
import numpy as np
import pandas as pd

ROOT = "/Users/cnavarreteliz/Documents/Repos/earthquakes"
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import models_and_evaluation as me  # noqa: E402  (imported, never modified)

FEAT_CSV = os.path.join(ROOT, "data", "processed", "chile_features_m40.csv")


def load_df_model(path=FEAT_CSV, start_year=1980.0):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    t_max_complete = df["time"].max() - pd.Timedelta(days=30)
    dm = df[(df["year_frac"] >= start_year) & (df["time"] <= t_max_complete)].copy().reset_index(drop=True)
    return df, dm


def m6_pathways(dm, target_col, train_mask, val_mask):
    """Exact replica of M6 (PI-ST-Fusion) and M2 from run_benchmark_for_target, but returning the
    three pathway z-scores so the hand-set fusion weights can be audited."""
    y_all = dm[target_col].values.astype(int)
    is_mega = (dm["time_since_last_m70_local"].values <= 60.0).astype(float)
    w_all = (1.0 - 0.92 * is_mega) / np.power(1.0 + dm["count_local_7d"].values, 0.85)
    from sklearn.preprocessing import RobustScaler
    import lightgbm as lgb
    import xgboost as xgb
    cols, mono = me.PI_ST_FUSION_COLS, me.PI_ST_FUSION_MONO
    sc = RobustScaler()
    Xr_tr = dm.loc[train_mask, cols].values
    Xr_all = dm[cols].values
    sc.fit(Xr_tr)
    Xs = np.clip(sc.transform(Xr_all), -4.0, 4.0) * np.array(mono)
    ci = {c: i for i, c in enumerate(cols)}
    s_m5 = (0.38 * Xs[:, ci["etas_aniso_intensity"]] + 0.24 * Xs[:, ci["etas_med_7d"]]
            + 0.18 * Xs[:, ci["cumulative_moment_release_10yr"]] + 0.12 * Xs[:, ci["background_rate_mu_local"]]
            + 0.08 * Xs[:, ci["frac_interface_30d"]])
    s_stress = (0.18 * Xs[:, ci["benioff_rate_7d"]] + 0.14 * Xs[:, ci["benioff_rate_30d"]]
                + 0.16 * Xs[:, ci["time_since_last_m60_local"]] + 0.14 * Xs[:, ci["delta_b_pos_25_100"]]
                + 0.08 * Xs[:, ci["b_pos_25"]] + 0.10 * Xs[:, ci["tectonic_slip_deficit_m"]]
                + 0.08 * Xs[:, ci["elastic_strain_index"]] + 0.06 * Xs[:, ci["strike_rad"]]
                + 0.06 * Xs[:, ci["mag"]])
    inter_tr = train_mask & (dm["time_since_last_m70_local"].values > 60.0)
    lg = lgb.LGBMClassifier(n_estimators=220, learning_rate=0.02, num_leaves=10, max_depth=3,
                            min_child_samples=60, subsample=0.8, colsample_bytree=0.8, reg_alpha=5.0,
                            reg_lambda=20.0, monotone_constraints=mono, random_state=42, n_jobs=1, verbose=-1)
    lg.fit(Xr_all[inter_tr], y_all[inter_tr], sample_weight=w_all[inter_tr])
    eps = 1e-5
    p = np.clip(lg.predict_proba(Xr_all)[:, 1], eps, 1 - eps)
    s_lgb = np.log(p / (1 - p))
    xg = xgb.XGBClassifier(n_estimators=220, learning_rate=0.02, max_depth=3, min_child_weight=35,
                           subsample=0.8, colsample_bytree=0.8, reg_alpha=5.0, reg_lambda=20.0,
                           monotone_constraints=tuple(mono), eval_metric="auc", random_state=42, n_jobs=1)
    xg.fit(Xr_all[inter_tr], y_all[inter_tr], sample_weight=w_all[inter_tr], verbose=False)
    p = np.clip(xg.predict_proba(Xr_all)[:, 1], eps, 1 - eps)
    s_xgb = np.log(p / (1 - p))

    def nz(a):
        m = np.average(a[inter_tr], weights=w_all[inter_tr])
        s = np.sqrt(np.average((a[inter_tr] - m) ** 2, weights=w_all[inter_tr])) + 1e-8
        return (a - m) / s

    z_m5, z_st, z_gb = nz(s_m5), nz(s_stress), 0.55 * nz(s_lgb) + 0.45 * nz(s_xgb)
    s_m2 = 0.80 * dm["etas_iso_intensity"].values + 0.20 * np.log1p(dm["background_rate_mu_local"].values * 365.0)
    return dict(y=y_all, w=w_all, z_m5=z_m5, z_stress=z_st, z_gbdt=z_gb, s_m2=s_m2, s_m5=s_m5)


def calibrate(score, y, w, train_mask, val_mask, test_mask):
    """The original calibration (including its slope-override hack)."""
    _, _, p_te, _ = me.map_score_to_calibrated_prob(score[train_mask], y[train_mask], w[train_mask],
                                                    score[val_mask], y[val_mask], w[val_mask], score[test_mask])
    return p_te


def bern_ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def block_bootstrap_ig(y, p_model, p_ref, years, n_boot=2000, seed=0):
    """IG per event (nats) with block bootstrap over calendar years. Returns (ig, lo95, hi95, p_one_sided)."""
    d = bern_ll(y, p_model) - bern_ll(y, p_ref)
    ig = float(d.mean())
    uy = np.unique(years)
    groups = [np.where(years == u)[0] for u in uy]
    sums = np.array([d[g].sum() for g in groups])
    ns = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        k = rng.integers(0, len(uy), len(uy))
        boots[b] = sums[k].sum() / ns[k].sum()
    return ig, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(np.mean(boots <= 0))


def block_bootstrap_auc_diff(y, p_a, p_b, years, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    uy = np.unique(years)
    groups = [np.where(years == u)[0] for u in uy]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(uy), len(uy))])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.mean(out <= 0))
