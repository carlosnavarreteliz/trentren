import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/station_noise")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import harness as H, features as F
z = pd.read_parquet(sys.argv[1])
p = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
f = F.panel_features(z, p).merge(p[["bin_id", "t_start", "split", "y_main", "y_all", "log_days_since_m6_nbr"]], on=["bin_id", "t_start"])
for sp in ("train", "test"):
    for y in ("y_main", "y_all"):
        for c in ("x_hf", "x_lp", "x_ms"):
            d = f[(f.split == sp) & f[c].notna()]
            print(sp, y, c, "n_pos", int(d[y].sum()), "AUC %.3f" % roc_auc_score(d[y], d[c]),
                  "mean z pos %.2f neg %.2f" % (d.loc[d[y] == 1, c].mean(), d.loc[d[y] == 0, c].mean()))
# site-level z in the 30 days before three large test-period mainshocks
for name, t, lat in [("Iquique M8.2", "2014-04-01", -19.6), ("Illapel M8.3", "2015-09-16", -31.6), ("Chiloe M7.6", "2016-12-25", -43.4)]:
    t = pd.Timestamp(t)
    zz = z[(np.abs(z.lat - lat) <= 2.5) & (z.date >= t - pd.Timedelta(days=30)) & (z.date < t)]
    print(name, zz.groupby(["sta", "band"]).z.mean().unstack().round(2).to_string())
