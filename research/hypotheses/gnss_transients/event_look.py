"""Event-level look at the feature before major Chilean earthquakes. usage: python -I event_look.py <station_anoms.parquet>"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, runpy
g = runpy.run_path("/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/gnss_transients/evaluate_gnss.py", run_name="lib")
W = g["WEEKS"]; G, C = g["bin_grid"](2, 30); G90, _ = g["bin_grid"](2, 90)
an = g["an"]; H = g["H"]
fin = G[np.isfinite(G)]
print("feature distribution (all bin-weeks with data, 2007+):", np.nanpercentile(G[W >= "2007-01-01"], [5, 25, 50, 75, 95, 99]).round(2))
EV = {"Tocopilla 2007-11-14 M7.7": ("2007-11-14", -22.25), "Maule 2010-02-27 M8.8": ("2010-02-27", -36.12),
      "Iquique 2014-04-01 M8.2": ("2014-04-01", -19.61), "Iquique foreshock 2014-03-16 M6.7": ("2014-03-16", -19.98),
      "Illapel 2015-09-16 M8.3": ("2015-09-16", -31.57), "Chiloe 2016-12-25 M7.6": ("2016-12-25", -43.42),
      "Valparaiso 2017-04-24 M6.9": ("2017-04-24", -33.04), "Coquimbo 2019-01-20 M6.7": ("2019-01-20", -30.04)}
for name, (d, lat) in EV.items():
    b = int(H.lat_to_bin(np.array([lat]))[0]); t = pd.Timestamp(d)
    k = np.searchsorted(W.values, np.datetime64(t), side="right") - 1
    rows = range(max(0, k - 11), k + 3)
    pct = lambda v: np.nan if not np.isfinite(v) else (fin < v).mean()
    print(f"\n{name}  bin {b}  (pct = percentile among all bin-weeks)")
    for i in rows:
        print(f"  {W[i].date()} {'<-- last issue before' if i == k else ('   after' if i > k else '')}  z30={G[i, b]:+.2f} (pct {pct(G[i,b]):.2f})  z90={G90[i, b]:+.2f}  nsta={int(C[i, b])}")
    # station-level, last issue before the event
    a = an[(an.lag == 2) & (an.t_start == W[k]) & (np.abs(an.bin - b) <= 1)]
    print("  stations at last issue:", a[["sta", "lat", "z30", "z90", "post65"]].round(2).to_string(index=False).replace("\n", "\n    "))
