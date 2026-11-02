"""Train-period (1976-2012) stand-alone comparison: NN out-of-fold vs base logit out-of-fold (same 5 time blocks,
26-week purge), level-matched. Checks whether the test-period stand-alone gain also exists before 2013.
Usage: python3 -I train_period_check.py WORK_DIR"""
import sys, os, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import brentq
import harness as H
WORK = sys.argv[1]; RES = os.path.join(os.path.dirname(WORK.rstrip("/")), "results")
nn = pd.read_parquet(os.path.join(WORK, "nn_preds.parquet")); nn["t_start"] = pd.to_datetime(nn.t_start).astype("datetime64[ns]")
P6 = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
sig = lambda z: 1 / (1 + np.exp(-z)); ll = lambda y, p: H._ll(y, np.clip(p, 1e-9, 1 - 1e-9))
out = {}
for target, head in (("y_main", "main6"), ("y_all", "all6")):
    b = H._baseline(P6, target, "logit")
    tr = b["tr"]; Xb = b["Xb"][tr]; y = b["y"][tr]
    pan = P6.loc[tr, ["bin_id", "t_start"]].reset_index(drop=True)
    t = pan.t_start.values.astype("datetime64[D]").astype(np.int64) // 7
    edges = np.quantile(t, np.linspace(0, 1, 6)); fold = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, 4)
    oof = np.zeros(len(y))
    for f in range(5):
        va = fold == f; lo, hi = t[va].min(), t[va].max()
        trm = (~va) & ((t < lo - 26) | (t > hi + 26))
        oof[va] = H._fit_predict(Xb[trm], y[trm], Xb[va], "logit")[0]
    pan["p_base_oof"], pan["y"] = oof, y
    m = pan.merge(nn, on=["bin_id", "t_start"]); yy = m.y.to_numpy(); N = yy.sum()
    zb = np.log(m.p_base_oof / (1 - m.p_base_oof)).to_numpy()
    for kind in ("gft", "co", "gzs"):
        z = m[f"{kind}_{head}"].to_numpy()
        c = brentq(lambda c: sig(z + c).sum() - sig(zb).sum(), -5, 5)
        m["p_lm"], m["year"] = sig(z + c), pd.to_datetime(m.t_start).dt.year
        out[f"{kind}_{target}"] = dict(n=int(N), raw_ig_vs_base_oof=float((ll(yy, sig(z)) - ll(yy, sig(zb))).sum() / N),
                                       levelmatched_ig=float((ll(yy, sig(z + c)) - ll(yy, sig(zb))).sum() / N),
                                       boot=H.block_bootstrap_ci(m, a="p_lm", b="p_base_oof"),
                                       note="gzs is not out-of-fold-trained on Chile (zero-shot) so it is genuinely out-of-sample")
        print(kind, target, out[f"{kind}_{target}"], flush=True)
json.dump(out, open(os.path.join(RES, "train_period_check.json"), "w"), indent=1)
