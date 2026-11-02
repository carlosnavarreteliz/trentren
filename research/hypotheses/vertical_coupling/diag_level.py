"""POST-HOC diagnostic: split IG into a 'level' part (test-period mean rate shift) and a 'shape' part.
Oracle step (diagnostic only, uses test outcomes): rescale each model's test log-odds intercept so that
sum(p) = observed count, then recompute IG. If IG vanishes after both are recalibrated, the gain was only a level
shift (compensating the known +32% test-period y_main rate increase), not discrimination.
Usage: python -I diag_level.py <out_dir>"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/vertical_coupling")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import brentq
import harness as H
out = sys.argv[1]
pw = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
F = pd.read_parquet(f"{out}/F1_weekly.parquet")
tr = pw.split == "train"; te = pw.split == "test"
M = pw[["bin_id", "t_start", "split"]].merge(F, on=["bin_id", "t_start"])
for c in ["x_ids_anom", "x_or_anom", "x_vert_shift"]:
    print(c, "train mean %.3f test mean %.3f" % (M[M.split == "train"][c].mean(), M[M.split == "test"][c].mean()))
ll = lambda y, q: np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
def recal(y, p):
    lo = np.log(p / (1 - p))
    s = brentq(lambda a: np.sum(1 / (1 + np.exp(-(lo + a)))) - y.sum(), -5, 5)
    return 1 / (1 + np.exp(-(lo + s)))
res = {}
for name, cols in [("F1", ["x_ids_anom", "x_or_anom", "x_vert_shift"]), ("x_or_anom", ["x_or_anom"]),
                   ("x_ids_anom", ["x_ids_anom"])]:
    for tgt in ("y_main", "y_all"):
        r = H.evaluate(F[["bin_id", "t_start"] + cols], target=tgt, panel=pw, return_preds=True)
        p = r["preds"]; y = p.y.to_numpy(); N = y.sum()
        ig_raw = (ll(y, p.p_full) - ll(y, p.p_base)) / N
        ig_rec = (ll(y, recal(y, p.p_full.to_numpy())) - ll(y, recal(y, p.p_base.to_numpy()))) / N
        d = dict(ig_raw=round(ig_raw, 4), ig_after_level_recal=round(ig_rec, 4), exp_base=round(p.p_base.sum(), 2),
                 exp_full=round(p.p_full.sum(), 2), obs=int(N))
        res[f"{name}|{tgt}"] = d; print(name, tgt, d)
json.dump(res, open(f"{out}/diag_level.json", "w"), indent=1, default=float)
