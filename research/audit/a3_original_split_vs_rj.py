"""A3 (secondary): on the ORIGINAL split (fit 1980-2017, test 2018+), compare the saved PI-ST-Fusion test
predictions against the generic Reasenberg-Jones clustering baseline (logistic fitted on 1980-2017)."""
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/audit")
from common import ROOT, load_df_model, block_bootstrap_ig, block_bootstrap_auc_diff, bern_ll  # noqa
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

df, dm = load_df_model(start_year=1980.0)
t_all, lat_all, lon_all, mag_all = df["t_days"].values, df["latitude"].values, df["longitude"].values, df["mag"].values
yk = lat_all * 111.19
xk = lon_all * 111.19 * np.cos(np.radians(lat_all))
pos = pd.Series(np.arange(len(df)), index=df["id"].values)
ix = pos.loc[dm["id"].values].values
a_rj, b_rj, p_rj, c_rj = -1.67, 0.91, 1.08, 0.05
I = lambda t0, t1: ((t0 + c_rj) ** (1 - p_rj) - (t1 + c_rj) ** (1 - p_rj)) / (p_rj - 1)
cand = np.where((mag_all >= 5.5) & (t_all >= 80 * 365.25) & (t_all < 118 * 365.25))[0]
N_rj, N_bg, N_rj_prior = np.zeros(len(dm)), np.zeros(len(dm)), np.zeros(len(dm))
for r, i in enumerate(ix):
    lo = np.searchsorted(t_all, t_all[i] - 365.0, "left")
    sl = slice(lo, i + 1)
    m = np.hypot(xk[sl] - xk[i], yk[sl] - yk[i]) <= 150.0
    tau = t_all[i] - t_all[sl][m]
    contrib = 10 ** (a_rj + b_rj * (mag_all[sl][m] - 5.5)) * I(tau, tau + 14.0)
    N_rj[r] = contrib.sum()
    N_rj_prior[r] = contrib[:-1].sum()  # EXCLUDING the trigger itself (what orig M2 effectively does)
    N_bg[r] = ((np.hypot(xk[cand] - xk[i], yk[cand] - yk[i]) <= 150).sum() + 0.5) / (38 * 365.25) * 14
yr = dm["year_frac"].values
fit, te = yr < 2018.0, yr >= 2018.0
y = dm["y_m55_14d"].values.astype(int)
X = np.column_stack([np.log(N_rj + 1e-5), np.log(N_bg)])
Xp = np.column_stack([np.log(N_rj_prior + 1e-5), np.log(N_bg)])
p_rj = LogisticRegression(C=1e3, max_iter=2000).fit(X[fit], y[fit]).predict_proba(X[te])[:, 1]
p_rjp = LogisticRegression(C=1e3, max_iter=2000).fit(Xp[fit], y[fit]).predict_proba(Xp[te])[:, 1]
tp = json.load(open(f"{ROOT}/results/test_predictions.json"))["Task1_M55_14d"]
yt = np.array(tp["y_te"])
assert np.array_equal(yt, y[te])
p6 = np.array(tp["preds_te"]["M6: PI-ST-Fusion (Ours)"])
p2 = np.array(tp["preds_te"]["M2: Classical-ETAS (Ogata 1998)"])
pre = np.array(tp["is_pre_mainshock"], bool)
years = np.floor(yr[te]).astype(int)
out = {"n_test": int(len(yt)), "n_pos": int(yt.sum())}
for k, p in {"M6": p6, "orig_M2": p2, "RJ_generic_incl_trigger": p_rj, "RJ_generic_excl_trigger": p_rjp}.items():
    out[k] = {"auc": roc_auc_score(yt, p), "prauc": average_precision_score(yt, p), "LL": float(bern_ll(yt, p).mean()),
              "auc_pre_mainshock_subset": roc_auc_score(yt[pre], p[pre])}
for a, b, pa, pb in [("M6", "RJ_generic_incl_trigger", p6, p_rj), ("M6", "orig_M2", p6, p2),
                     ("RJ_generic_incl_trigger", "RJ_generic_excl_trigger", p_rj, p_rjp)]:
    ig, lo, hi, pv = block_bootstrap_ig(yt, pa, pb, years)
    alo, ahi, apv = block_bootstrap_auc_diff(yt, pa, pb, years, n_boot=500)
    out[f"{a}_vs_{b}"] = {"IG_per_event": ig, "IG_per_target": ig * len(yt) / yt.sum(), "CI95": [lo, hi], "p": pv,
                          "dAUC": roc_auc_score(yt, pa) - roc_auc_score(yt, pb), "dAUC_CI95": [alo, ahi], "p_dAUC": apv}
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(f"{ROOT}/research/audit/a3_results.json", "w"), indent=1, default=float)
