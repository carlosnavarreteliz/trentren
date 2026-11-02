"""A1: reproduce the published Task-1 metric, test whether the hand-set fusion weights look test-tuned,
and quantify how much of the signal is the triggering event's own magnitude / clustering."""
import json
import sys
import numpy as np
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/audit")
from common import ROOT, load_df_model, m6_pathways, calibrate, me  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

out = {}
# 1. Re-score the saved test predictions
tp = json.load(open(f"{ROOT}/results/test_predictions.json"))["Task1_M55_14d"]
y = np.array(tp["y_te"])
for k, v in tp["preds_te"].items():
    out.setdefault("saved_preds_auc", {})[k] = roc_auc_score(y, v)

# 2. Re-run M6 exactly on the original split (1980-2013 / 2014-2017 / 2018+)
df, dm = load_df_model()
yr = dm["year_frac"].values
tr, va, te = yr < 2014, (yr >= 2014) & (yr < 2018), yr >= 2018
P = m6_pathways(dm, "y_m55_14d", tr, va)
yy, w = P["y"], P["w"]
s6 = 0.48 * P["z_m5"] + 0.34 * P["z_stress"] + 0.18 * P["z_gbdt"]
p6 = calibrate(s6, yy, w, tr, va, te)
p2 = calibrate(P["s_m2"], yy, w, tr, va, te)
out["rerun_m6_test_auc"] = roc_auc_score(yy[te], p6)
out["rerun_m2_test_auc"] = roc_auc_score(yy[te], p2)
out["n_test"], out["n_test_pos"] = int(te.sum()), int(yy[te].sum())

# 3. Single-feature AUCs on test (no fitting at all)
mag = dm["mag"].values
for f in ["mag", "etas_iso_intensity", "etas_aniso_intensity", "max_mag_local_7d", "count_local_7d"]:
    out.setdefault("single_feature_test_auc", {})[f] = roc_auc_score(yy[te], dm[f].values[te])
# "ETAS + own magnitude": a 2-feature logistic fitted on train only
X = np.column_stack([mag, dm["etas_iso_intensity"].values,
                     np.log1p(dm["background_rate_mu_local"].values * 365)])
lr = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr | va], yy[tr | va])
pe = lr.predict_proba(X[te])[:, 1]
out["etas_plus_own_mag_logit_test_auc"] = roc_auc_score(yy[te], pe)
out["etas_plus_own_mag_logit_test_prauc"] = average_precision_score(yy[te], pe)
out["m6_test_prauc"] = average_precision_score(yy[te], p6)

# 4. Is the "pre-mainshock" regime declustered?  fraction of events kept, and what share of
#    positive test labels are produced by an event that is itself M>=5.5 or near a recent M>=5.5
pre = dm["is_pre_mainshock_regime"].values.astype(bool)
out["pre_mainshock_frac_of_test_kept"] = float(pre[te].mean())
out["pre_mainshock_frac_of_test_positives_kept"] = float(pre[te & (yy == 1)].mean())
out["test_positive_rate_if_trigger_M>=5.5"] = float(yy[te & (mag >= 5.5)].mean())
out["test_positive_rate_if_trigger_M<4.5"] = float(yy[te & (mag < 4.5)].mean())
out["share_of_test_positives_with_target_within_1d_of_bigger_prior"] = None  # see a2

# 5. Hand-set fusion weight audit: rank of (0.48,0.34,0.18) among random simplex weights,
#    on validation (allowed) vs test (forbidden) AUC.
rng = np.random.default_rng(1)
W = rng.dirichlet([1, 1, 1], 2000)
Z = np.column_stack([P["z_m5"], P["z_stress"], P["z_gbdt"]])
chosen = np.array([0.48, 0.34, 0.18])
def aucs(wv):
    s = Z @ wv
    return roc_auc_score(yy[va], s[va]), roc_auc_score(yy[te], s[te])
ch_va, ch_te = aucs(chosen)
rv = np.array([aucs(wv) for wv in W])
out["fusion_weights"] = {
    "chosen_val_auc": ch_va, "chosen_test_auc": ch_te,
    "pct_random_weights_beating_chosen_on_val": float(np.mean(rv[:, 0] > ch_va)),
    "pct_random_weights_beating_chosen_on_test": float(np.mean(rv[:, 1] > ch_te)),
    "best_val_weights": W[np.argmax(rv[:, 0])].round(3).tolist(),
    "test_auc_of_best_val_weights": float(rv[np.argmax(rv[:, 0]), 1]),
    "single_pathway_test_auc": {k: roc_auc_score(yy[te], P[k][te]) for k in ["z_m5", "z_stress", "z_gbdt"]},
    "single_pathway_val_auc": {k: roc_auc_score(yy[va], P[k][va]) for k in ["z_m5", "z_stress", "z_gbdt"]},
}

# 6. Calibration hack: does the slope override fire?
_, _, _, (ms, ss, cal) = me.map_score_to_calibrated_prob(s6[tr], yy[tr], w[tr], s6[va], yy[va], w[va], s6[te])
out["m6_cal_slope"] = float(cal.coef_[0, 0])
out["m6_test_mean_pred_vs_obs"] = [float(p6.mean()), float(yy[te].mean())]
# reliability by decile
q = np.quantile(p6, np.linspace(0, 1, 11))
rel = []
for a, b in zip(q[:-1], q[1:]):
    m = (p6 >= a) & (p6 <= b)
    rel.append([round(float(p6[m].mean()), 4), round(float(yy[te][m].mean()), 4), int(m.sum())])
out["m6_test_reliability_deciles[pred,obs,n]"] = rel

print(json.dumps(out, indent=1, default=float))
json.dump(out, open(f"{ROOT}/research/audit/a1_results.json", "w"), indent=1, default=float)
