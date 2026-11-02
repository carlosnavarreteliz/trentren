"""Exploratory diagnostics of the H1 gain (after the preregistered primary test). Args: cat1 cat2"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H
from tele_features import load_global, TeleFeatures
panel = H.get_panel(7, 7, 6.0)
tf = TeleFeatures(load_global(sys.argv[1:3]))
f = tf.h1(panel)
def ll(y, p): return y*np.log(np.clip(p,1e-12,1)) + (1-y)*np.log(np.clip(1-p,1e-12,1))
for target in ("y_all", "y_main"):
    r = H.evaluate(f, target=target, panel=panel, return_preds=True)
    print(target, "coef", r["coef_extra"], "IG", round(r["ig_per_event"],4))
    for c in ["x_strain_max_7d", "x_strain_sum_30d"]:
        rc = H.evaluate(f[["t_start","bin_id",c]], target=target, panel=panel)
        print("   single", c, round(rc["ig_per_event"],4), rc["coef_extra"])
    pr = r["preds"].merge(f, on=["bin_id","t_start"])
    pr["d"] = ll(pr.y, pr.p_full) - ll(pr.y, pr.p_base)
    pos = pr[pr.y == 1]
    print("   sum d on positives %.3f, on negatives %.3f" % (pos.d.sum(), pr[pr.y==0].d.sum()))
    print("   ratio p_full/p_base: positives median %.3f mean %.3f; all cells mean %.3f" % (
        (pos.p_full/pos.p_base).median(), (pos.p_full/pos.p_base).mean(), (pr.p_full/pr.p_base).mean()))
    print(pos.sort_values("d")[["bin_id","t_start","p_base","p_full","d","x_strain_max_7d","x_strain_sum_30d"]].tail(8).to_string())
    print(pos.sort_values("d")[["bin_id","t_start","p_base","p_full","d","x_strain_max_7d","x_strain_sum_30d"]].head(4).to_string())
    # binned: positive rate vs feature in test, and in train
    for c in ["x_strain_sum_30d"]:
        q = pd.qcut(panel[c] if c in panel else pr[c], 4, duplicates="drop")
    tr = panel[panel.split=="train"][["bin_id","t_start",target,"p_poisson_all"]].merge(f, on=["bin_id","t_start"])
    te = panel[panel.split=="test"][["bin_id","t_start",target,"p_poisson_all"]].merge(f, on=["bin_id","t_start"])
    for name, d in (("train", tr), ("test", te)):
        d["q"] = pd.cut(d.x_strain_sum_30d, [-4, -2.5, -1, 0, 1, 5])
        g = d.groupby("q", observed=True).agg(n=(target,"size"), obs=(target,"sum"), exp=("p_poisson_all","sum"))
        g["obs/exp_poisson"] = (g.obs/g.exp).round(2)
        print("  ", name, "\n", g.to_string())
