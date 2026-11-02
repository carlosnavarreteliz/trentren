"""Diagnostic figure. Usage: python -I make_figure.py <ipoc.txt>"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research"); sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import harness as H, ec_load, ec_features as F
OUT = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog/"
cat = ec_load.load_ipoc(sys.argv[1])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
y = cat.groupby(cat.time.dt.year).mag
for thr, lab in [(None, "todos"), (2.5, "ML>=2.5"), (3.0, "ML>=3.0"), (4.5, "ML>=4.5")]:
    s = y.size() if thr is None else y.apply(lambda m: (m >= thr).sum())
    ax[0].semilogy(s.index, s.values, marker="o", label=lab)
ax[0].set_title("IPOC: eventos por año"); ax[0].legend()
P, ob = F.sub_panel(7, 7, 5.0, 22, 27)
f = F.build_features(cat, P, ob, 3.0, 365, "2008-01-01")
d = P[["bin_id", "t_start", "y_main", "split"]].merge(f, on=["bin_id", "t_start"])
d = d[d.t_start >= "2008-01-01"]
for k, (c, lab) in enumerate([("x_act7", "x_act7 (activación 7 d, ML>=3)"), ("x_dbpos", "x_dbpos (Δ b-positivo)")]):
    a = ax[k + 1]
    for yv, col in [(0, "tab:gray"), (1, "tab:red")]:
        v = d[(d.y_main == yv)][c].dropna()
        a.hist(v, bins=40, density=True, alpha=0.5, color=col, label=f"y_main={yv} (n={len(v)})")
    a.set_title(lab + ", 2008-2021"); a.legend()
plt.tight_layout(); plt.savefig(OUT + "fig_diagnostics.png", dpi=110)
print("ok")
