"""Exploratory: is the H1 gain a level shift (global quiet in the test period vs Chile active)? Args: cat1 cat2"""
import sys, json, glob
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H
from tele_features import load_global, TeleFeatures
tf = TeleFeatures(load_global(sys.argv[1:3]))
g = tf.g.copy(); g["yr"] = g.time.dt.year; g["rem"] = tf.remote_all; g["sh"] = g.depth <= 100
r = g[g.rem & g.sh]
print("remote shallow M>=6.5 per yr: 1976-2012 %.1f, 2013-2026 %.1f" % (
    len(r[(r.yr>=1976)&(r.yr<=2012)])/37, len(r[r.yr>=2013])/(2026.8-2013)))
panel = H.get_panel(7, 7, 6.0)
f = tf.h1(panel).merge(panel[["bin_id","t_start","split"]])
print(f.groupby("split")[["x_strain_max_7d","x_strain_sum_30d"]].mean())
# global-broadcast variant (does geometry matter?)
gl = f.groupby("t_start")[["x_strain_max_7d","x_strain_sum_30d"]].max().reset_index()
gl.columns = ["t_start","x_gmax7","x_gsum30"]
for t in ("y_all","y_main"):
    rr = H.evaluate(gl, target=t, panel=panel); print(t, "global-broadcast IG", round(rr["ig_per_event"],4), rr["coef_extra"])
# indicator: any remote shallow M>=6.5 in last 7 d
ind = gl.assign(x_any7=(gl.x_gmax7 > -3).astype(float))[["t_start","x_any7"]]
for t in ("y_all","y_main"):
    rr = H.evaluate(ind, target=t, panel=panel); print(t, "indicator any-remote-7d IG", round(rr["ig_per_event"],4), rr["coef_extra"])
for fn in sorted(glob.glob("results/h1_y_*_s7_m6.0_d1000_logit.json")):
    d = json.load(open(fn)); print(fn, "per-year dLL", d["per_year_dll"]); print("  LOYO min/max", min(v for v in d["loyo_ig"].values()), max(d["loyo_ig"].values()))
