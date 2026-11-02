"""A2 (pre-registered primary test): PI-ST-Fusion vs a generic Reasenberg-Jones/ETAS clustering baseline
on the mandated split (fit 1976-2012, test 2013+), on the raw and the Gardner-Knopoff declustered target,
plus the incremental value of 'physics' features over clustering-only features."""
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/audit")
from common import ROOT, load_df_model, m6_pathways, calibrate, block_bootstrap_ig, block_bootstrap_auc_diff, bern_ll  # noqa
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

TEST_START = float(sys.argv[1]) if len(sys.argv) > 1 else 2013.0
START = float(sys.argv[2]) if len(sys.argv) > 2 else 1976.0
TAG = "" if TEST_START == 2013.0 else f"_test{int(TEST_START)}"
df, dm = load_df_model(start_year=START)
# full M>=4 catalog (1900+) for parents / declustering
t_all = df["t_days"].values
lat_all, lon_all, mag_all = df["latitude"].values, df["longitude"].values, df["mag"].values
yk = lat_all * 111.19
xk = lon_all * 111.19 * np.cos(np.radians(lat_all))

# ---------- Gardner-Knopoff (1974) declustering of the M>=4 catalog ----------
def gk_window(m):
    L = 10 ** (0.1238 * m + 0.983)
    T = np.where(m >= 6.5, 10 ** (0.032 * m + 2.7389), 10 ** (0.5409 * m - 0.547))
    return L, T
dep = np.zeros(len(df), bool)
L_all, T_all = gk_window(mag_all)
for i in np.argsort(-mag_all, kind="stable"):
    if dep[i] or mag_all[i] < 4.5:
        continue
    lo = np.searchsorted(t_all, t_all[i] - T_all[i] * 0.0, "left")  # aftershocks
    hi = np.searchsorted(t_all, t_all[i] + T_all[i], "right")
    lo_f = np.searchsorted(t_all, t_all[i] - T_all[i], "left")      # foreshocks (same window)
    for a, b in [(lo_f, hi)]:
        d = np.hypot(xk[a:b] - xk[i], yk[a:b] - yk[i])
        sel = np.arange(a, b)[(d <= L_all[i]) & (mag_all[a:b] <= mag_all[i])]
        sel = sel[sel != i]
        dep[sel] = True
mainshock = ~dep
df["gk_mainshock"] = mainshock

# map dm rows to df rows by id
pos = pd.Series(np.arange(len(df)), index=df["id"].values)
ix = pos.loc[dm["id"].values].values

# ---------- Reasenberg-Jones generic expected count & declustered target ----------
a_rj, b_rj, p_rj, c_rj = -1.67, 0.91, 1.08, 0.05
def rj_integral(t0, t1):
    return ((t0 + c_rj) ** (1 - p_rj) - (t1 + c_rj) ** (1 - p_rj)) / (p_rj - 1)
n = len(dm)
N_rj = np.zeros(n)
y_dc = np.zeros(n, int)
for r in range(n):
    i = ix[r]
    ti = t_all[i]
    lo = np.searchsorted(t_all, ti - 365.0, "left")
    sl = slice(lo, i + 1)  # includes the trigger itself
    d = np.hypot(xk[sl] - xk[i], yk[sl] - yk[i])
    m = d <= 150.0
    tau = ti - t_all[sl][m]
    N_rj[r] = np.sum(10 ** (a_rj + b_rj * (mag_all[sl][m] - 5.5)) * rj_integral(tau, tau + 14.0))
    hi = np.searchsorted(t_all, ti + 14.0, "right")
    fs = slice(i + 1, hi)
    d2 = np.hypot(xk[fs] - xk[i], yk[fs] - yk[i])
    y_dc[r] = int(np.any((d2 <= 150.0) & (mag_all[fs] >= 5.5) & mainshock[fs]))
dm["y_dc"] = y_dc
dm["log_N_rj"] = np.log(N_rj + 1e-5)

yr = dm["year_frac"].values
fit = yr < TEST_START
tr, va, te = yr < TEST_START - 4, (yr >= TEST_START - 4) & (yr < TEST_START), yr >= TEST_START
years_te = np.floor(yr[te]).astype(int)

# static background: rate of GK-mainshock M>=5.5 within 150 km per 14 d, 1976-2012 only
cand = np.where((mag_all >= 5.5) & (t_all >= (1976 - 1900) * 365.25) & (t_all < (TEST_START - 1900) * 365.25))[0]
span = (TEST_START - 1976) * 365.25
N_bg = np.zeros(n); N_bg_dc = np.zeros(n)
for r in range(n):
    i = ix[r]
    d = np.hypot(xk[cand] - xk[i], yk[cand] - yk[i]) <= 150.0
    N_bg[r] = (d.sum() + 0.5) / span * 14.0
    N_bg_dc[r] = ((d & mainshock[cand]).sum() + 0.5) / span * 14.0
dm["log_N_bg"], dm["log_N_bg_dc"] = np.log(N_bg), np.log(N_bg_dc)

CLUST = ["mag", "log_N_rj", "log_N_bg", "etas_iso_intensity", "etas_aniso_intensity", "etas_short_1d",
         "etas_med_7d", "etas_long_30d", "background_rate_mu_local", "count_local_1d", "count_local_7d",
         "count_local_30d", "count_local_90d", "count_local_365d", "rate_ratio_1d_30d", "rate_ratio_7d_90d",
         "rate_ratio_30d_365d", "max_mag_local_7d", "max_mag_local_30d", "max_mag_local_365d",
         "time_since_last_m50_local", "time_since_last_m60_local", "time_since_last_m70_local",
         "benioff_rate_7d", "benioff_rate_30d", "interevent_dt_log_sec", "interevent_dist_km",
         "time_for_last_20_events_days"]
PHYS = ["b_aki_50", "b_pos_25", "b_pos_50", "b_pos_100", "b_pos_250", "delta_b_pos_25_100", "delta_b_pos_50_250",
        "stai_bias_aki_minus_bpos", "gulia_wiemer_tl_score", "mag_deficit_bath", "b_pos_subthresh_40",
        "delta_b_pos_subthresh", "frac_interface_30d", "frac_intraslab_30d", "intraslab_to_interface_ratio_30d",
        "intraslab_rate_change_14d_90d", "mean_depth_local_30d", "depth_migration_vel_km_day",
        "along_strike_spread_km_30d", "trench_migration_km", "locking_gradient_migration",
        "years_since_giant_rupture", "tectonic_slip_deficit_m", "elastic_strain_index",
        "cumulative_moment_release_10yr", "zaliapin_log_eta", "zaliapin_log_T", "zaliapin_log_R",
        "spatial_entropy_30d", "z_rel_km", "d_trench_km", "depth", "locking_chi", "strike_rad",
        "omori_deviation_residual", "benioff_accel_7d_30d", "mean_mag_local_50", "std_mag_local_50"]
LOC = ["latitude", "longitude", "depth"]

def gbdt(cols, y):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0,
                           random_state=0, n_jobs=2, verbose=-1)
    m.fit(dm.loc[fit, cols].values, y[fit])
    return m.predict_proba(dm.loc[te, cols].values)[:, 1]

results = {}
for tname, ycol, bgcol in [("raw_y_m55_14d", "y_m55_14d", "log_N_bg"), ("declustered_y_dc", "y_dc", "log_N_bg_dc")]:
    y = dm[ycol].values.astype(int)
    yt = y[te]
    # RJ/ETAS baseline (2-parameter logistic on train)
    Xb = dm[["log_N_rj", bgcol]].values
    p_rj = LogisticRegression(C=1e3, max_iter=2000).fit(Xb[fit], y[fit]).predict_proba(Xb[te])[:, 1]
    p_pois = LogisticRegression(C=1e3, max_iter=2000).fit(Xb[fit][:, 1:], y[fit]).predict_proba(Xb[te][:, 1:])[:, 1]
    # PI-ST-Fusion, re-fitted on the mandated split with its own procedure
    P = m6_pathways(dm.assign(**{ycol: y}), ycol, tr, va)
    s6 = 0.48 * P["z_m5"] + 0.34 * P["z_stress"] + 0.18 * P["z_gbdt"]
    p6 = calibrate(s6, y, P["w"], tr, va, te)
    p2 = calibrate(P["s_m2"], y, P["w"], tr, va, te)
    # GBDT ablations
    p_c = gbdt(CLUST, y)
    p_cl = gbdt(CLUST + LOC, y)
    p_cp = gbdt(CLUST + PHYS, y)
    preds = {"Poisson_static": p_pois, "RJ_ETAS_generic": p_rj, "orig_M2_ETAS": p2, "PI_ST_Fusion_M6": p6,
             "GBDT_clustering": p_c, "GBDT_clust+latlondepth": p_cl, "GBDT_clust+physics": p_cp}
    R = {"n_test": int(te.sum()), "n_pos": int(yt.sum()), "base_rate": float(yt.mean())}
    for k, p in preds.items():
        R[k] = {"auc": roc_auc_score(yt, p), "prauc": average_precision_score(yt, p),
                "mean_pred": float(p.mean()), "LL_per_event": float(bern_ll(yt, p).mean())}
    comps = [("PI_ST_Fusion_M6", "RJ_ETAS_generic"), ("PI_ST_Fusion_M6", "Poisson_static"),
             ("RJ_ETAS_generic", "Poisson_static"), ("orig_M2_ETAS", "RJ_ETAS_generic"),
             ("GBDT_clustering", "RJ_ETAS_generic"), ("GBDT_clust+physics", "GBDT_clustering"),
             ("GBDT_clust+physics", "GBDT_clust+latlondepth"), ("GBDT_clust+latlondepth", "GBDT_clustering")]
    for a, b in comps:
        ig, lo, hi, pv = block_bootstrap_ig(yt, preds[a], preds[b], years_te)
        alo, ahi, apv = block_bootstrap_auc_diff(yt, preds[a], preds[b], years_te, n_boot=500)
        R[f"{a}_vs_{b}"] = {"IG_nats_per_event": ig, "IG_nats_per_target": ig * len(yt) / max(1, yt.sum()),
                            "IG_CI95": [lo, hi], "p_one_sided": pv,
                            "dAUC": R[a]["auc"] - R[b]["auc"], "dAUC_CI95": [alo, ahi], "p_dAUC": apv}
    results[tname] = R
    print(tname, json.dumps(R, indent=1, default=float))

# subset: trigger events that are themselves GK mainshocks (removes aftershock triggers)
results["frac_test_triggers_that_are_gk_mainshocks"] = float(mainshock[ix][te].mean())
results["frac_raw_positives_that_are_dc_positives"] = float(
    (dm["y_dc"].values[te] & dm["y_m55_14d"].values[te]).sum() / max(1, dm["y_m55_14d"].values[te].sum()))
json.dump(results, open(f"{ROOT}/research/audit/a2_results{TAG}.json", "w"), indent=1, default=float)
print({k: v for k, v in results.items() if not isinstance(v, dict)})
