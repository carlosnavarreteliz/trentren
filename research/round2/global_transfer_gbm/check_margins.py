"""Sanity check of margin assignment: counts, perpendicular-distance histograms, map. Usage: check_margins.py <raw_dir> <out_png>"""
import sys, os
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import margins as MG
from build_panels import load_global  # noqa (main not executed)

g = load_global(sys.argv[1])
code, s, d, signs = MG.assign(g.longitude.to_numpy(), g.latitude.to_numpy(), g.depth.to_numpy())
g["margin"], g["d"] = code, d
print("total", len(g), "assigned", (code != "").sum(), "chile box", MG.in_chile_box(g.longitude, g.latitude).sum())
print("signs", signs)
big = g[g.mag >= 7.0]
print("M7+ global", len(big), "in margins", (big.margin != "").sum(), "in Chile box",
      MG.in_chile_box(big.longitude, big.latitude).sum())
print(big[(big.margin == "") & ~MG.in_chile_box(big.longitude, big.latitude)][["time", "latitude", "longitude", "mag"]]
      .assign(lo=lambda x: x.longitude.round(0), la=lambda x: x.latitude.round(0)).groupby(["la", "lo"]).size()
      .sort_values(ascending=False).head(25))
for m in MG.MARGINS:
    e = g[g.margin == m]
    print(m, len(e), "M6+", (e.mag >= 6).sum(), "M7+", (e.mag >= 7).sum(), "d pct", np.percentile(e.d, [5, 50, 95]).round(0))
fig, ax = plt.subplots(figsize=(18, 9))
lon360 = g.longitude % 360
ax.scatter(lon360[code == ""], g.latitude[code == ""], s=0.3, c="0.8")
for m in MG.MARGINS:
    sel = code == m
    ax.scatter(lon360[sel], g.latitude[sel], s=0.3)
    P = np.array(MG.MARGINS[m]); ax.plot(P[:, 0] % 360, P[:, 1], "k-", lw=1); ax.text(P[0, 0] % 360, P[0, 1], m)
ch = MG.in_chile_box(g.longitude, g.latitude)
ax.scatter(lon360[ch], g.latitude[ch], s=0.3, c="r")
fig.savefig(os.path.join(sys.argv[2], "margins_map.png"), dpi=110)
