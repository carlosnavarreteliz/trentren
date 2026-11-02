"""Exploratory: H2 y_main -- coefficients, train/test levels, single features, per-year. Args: cat1 cat2"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H
from tele_features import load_global, TeleFeatures
tf = TeleFeatures(load_global(sys.argv[1:3]))
panel = H.get_panel(7, 7, 6.0)
f = tf.h2(panel)
sp = panel[panel.bin_id == 0][["t_start", "split"]]
m = f.merge(sp)
print(m.groupby("split")[["x_gmoment_365d", "x_pacific_m7_365d"]].agg(["mean", "std"]))
print(m.assign(y=m.t_start.dt.year).groupby("y")[["x_gmoment_365d", "x_pacific_m7_365d"]].mean().round(2).T.to_string())
for t in ("y_all", "y_main"):
    r = H.evaluate(f, target=t, panel=panel); print(t, "both", round(r["ig_per_event"], 4), r["coef_extra"])
    for c in ["x_gmoment_365d", "x_pacific_m7_365d"]:
        rc = H.evaluate(f[["t_start", c]], target=t, panel=panel); print("   ", c, round(rc["ig_per_event"], 4), rc["coef_extra"])
d = json.load(open("results/h2_y_main_s7_m6.0_d1000_logit.json"))
print("per-year", d["per_year_dll"]); print("LOYO", {k: round(v, 3) for k, v in d["loyo_ig"].items()})
# yearly Chile mainshock cells vs pacific m7 count
pm = panel.groupby("year").agg(ymain=("y_main", "sum"))
pm["pac7"] = m.assign(year=m.t_start.dt.year).groupby("year").x_pacific_m7_365d.mean()
print("corr yearly ymain cells vs pacific m7 (1976-2012):", round(pm.loc[1976:2012].corr().iloc[0, 1], 3),
      " (2013-2025):", round(pm.loc[2013:2025].corr().iloc[0, 1], 3))
