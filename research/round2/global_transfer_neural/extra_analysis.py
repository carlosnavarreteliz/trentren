"""Post-hoc leakage/level checks of the stand-alone NN gain (not preregistered; descriptive).
Usage: python3 -I extra_analysis.py WORK_DIR  -> ../results/extra.json"""
import sys, os, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score
import harness as H
WORK = sys.argv[1]; RES = os.path.join(os.path.dirname(WORK.rstrip("/")), "results")
nn = pd.read_parquet(os.path.join(WORK, "nn_preds.parquet")); nn["t_start"] = pd.to_datetime(nn.t_start).astype("datetime64[ns]")
P6 = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
sig = lambda z: 1 / (1 + np.exp(-z))
ll = lambda y, p: H._ll(y, np.clip(p, 1e-9, 1 - 1e-9))
out = {}
for target, head in (("y_main", "main6"), ("y_all", "all6")):
    r = H.evaluate(None, target=target, model="logit", panel=P6, return_preds=True)["preds"]
    m = r.merge(nn, on=["bin_id", "t_start"])
    y = m.y.to_numpy(); N = y.sum()
    zb = np.log(m.p_base / (1 - m.p_base)).to_numpy()
    for kind in ("gft", "co", "gzs"):
        z = m[f"{kind}_{head}"].to_numpy()
        res = {}
        # (a) as is
        res["raw_ig_vs_base"] = float((ll(y, sig(z)) - ll(y, sig(zb))).sum() / N)
        # (b) NN level shifted so its test total equals the base test total (removes level difference)
        c = brentq(lambda c: sig(z + c).sum() - sig(zb).sum(), -5, 5)
        res["levelmatched_to_base_ig"] = float((ll(y, sig(z + c)) - ll(y, sig(zb))).sum() / N)
        # (c) both models shifted to the observed total (oracle level for both)
        cn = brentq(lambda c: sig(z + c).sum() - N, -5, 5); cb = brentq(lambda c: sig(zb + c).sum() - N, -5, 5)
        res["both_oracle_level_ig"] = float((ll(y, sig(z + cn)) - ll(y, sig(zb + cb))).sum() / N)
        m["p_lm"] = sig(z + c)
        res["boot_levelmatched"] = H.block_bootstrap_ci(m, a="p_lm", b="p_base")
        res["sum_p_nn"], res["sum_p_base"], res["observed"] = float(sig(z).sum()), float(sig(zb).sum()), int(N)
        res["auc_nn"], res["auc_base"] = float(roc_auc_score(y, z)), float(roc_auc_score(y, zb))
        # discrimination excluding cells with a recent M>=6 nearby (non-cluster cells): AUC on y within low-ETAS cells
        q = m.p_etas.to_numpy() < np.quantile(m.p_etas, 0.9)
        res["auc_nn_low_etas90"], res["auc_base_low_etas90"] = float(roc_auc_score(y[q], z[q])), float(roc_auc_score(y[q], zb[q]))
        res["corr_nn_base_logit"] = float(np.corrcoef(z, zb)[0, 1])
        out[f"{kind}_{target}"] = res
        print(kind, target, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "boot_levelmatched"},
              {k: round(v, 3) for k, v in res["boot_levelmatched"].items()}, flush=True)
json.dump(out, open(os.path.join(RES, "extra.json"), "w"), indent=1)
