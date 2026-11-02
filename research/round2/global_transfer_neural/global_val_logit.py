"""Diagnostic (pretraining data only, <2013, non-Chile): does the GRU beat a logistic regression on the same 22
harness-style features on the held-out (margin, 2-yr) validation units?  Usage: python3 -I global_val_logit.py WORK"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
WORK = sys.argv[1]
g = np.load(os.path.join(WORK, "global_samples.npz"))
X, Y, K = np.nan_to_num(g["X"]), g["Y"], g["key"]
rng0 = np.random.default_rng(12345)
year = (pd.to_datetime(g["t_starts"]).year.values)[K[:, 1]]
unit = K[:, 0] * 100 + (year - 1976) // 2
units = np.unique(unit)
val_units = rng0.choice(units, size=int(0.2 * len(units)), replace=False)
va = np.isin(unit, val_units)
mu, sd = X[~va].mean(0), X[~va].std(0) + 1e-6
Z = (X - mu) / sd
out = {}
for j, h in enumerate(["all55", "main55", "all6", "main6", "all7"]):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Z[~va], Y[~va, j])
    p = np.clip(m.predict_proba(Z[va])[:, 1], 1e-9, 1 - 1e-9)
    y = Y[va, j]
    out[h] = dict(logit_val_ll=float((y * np.log(p) + (1 - y) * np.log1p(-p)).sum()), n_pos=int(y.sum()))
pl = json.load(open(os.path.join(WORK, "pretrain_log.json")))
for j, h in enumerate(out):
    nn_ll = np.mean([max(hh, key=lambda x: x["score"])["val_ll"][j] for hh in pl["hist"].values()])
    out[h]["gru_val_ll_mean_seeds"] = float(nn_ll)
    out[h]["const_val_ll"] = pl["const_val_ll"][j]
    out[h]["ig_gru_vs_logit_per_event"] = (nn_ll - out[h]["logit_val_ll"]) / out[h]["n_pos"]
    out[h]["ig_logit_vs_const_per_event"] = (out[h]["logit_val_ll"] - out[h]["const_val_ll"]) / out[h]["n_pos"]
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(WORK, "global_val_logit.json"), "w"), indent=1)
