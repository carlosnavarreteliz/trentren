"""Per-year and drop-one-year breakdown of the stand-alone (level-matched) NN-vs-base gain on test, and an
'is it carried by large sequences' split (cells with ETAS prob in the top decile vs the rest).
Usage: python3 -I per_year_alone.py WORK_DIR"""
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
    m = H.evaluate(None, target=target, model="logit", panel=P6, return_preds=True)["preds"].merge(nn, on=["bin_id", "t_start"])
    y = m.y.to_numpy(); zb = np.log(m.p_base / (1 - m.p_base)).to_numpy()
    for kind in ("gft", "gzs"):
        z = m[f"{kind}_{head}"].to_numpy(); c = brentq(lambda c: sig(z + c).sum() - sig(zb).sum(), -5, 5)
        d = ll(y, sig(z + c)) - ll(y, sig(zb))
        g = pd.DataFrame({"year": m.year, "d": d, "y": y}).groupby("year").sum()
        hi = m.p_etas.to_numpy() >= np.quantile(m.p_etas, 0.9)
        out[f"{kind}_{target}"] = dict(
            per_year={int(k): [round(float(r.d), 3), int(r.y)] for k, r in g.iterrows()},
            drop_one_year_min=float(min((g.d.sum() - g.d[k]) / (g.y.sum() - g.y[k]) for k in g.index)),
            dLL_top10pct_etas_cells=[round(float(d[hi].sum()), 3), int(y[hi].sum())],
            dLL_other_cells=[round(float(d[~hi].sum()), 3), int(y[~hi].sum())])
        print(kind, target, out[f"{kind}_{target}"], flush=True)
json.dump(out, open(os.path.join(RES, "per_year_alone.json"), "w"), indent=1)
