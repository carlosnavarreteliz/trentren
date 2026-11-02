"""Descriptive event-level check of the Bouchon et al. (2016) claim: x_ids_anom (and the other F1 features) at the
last weekly issue time before each large interface-type mainshock, vs. that bin's own distribution of the feature.
Usage: python -I event_level.py <out_dir>"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/vertical_coupling")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H, vc_features as V

out = sys.argv[1]
F = pd.read_parquet(f"{out}/F1_weekly.parquet")
ev = V.classify(H.get_events(min_mag=6.5, declustered=True))
ev = ev[ev.time >= pd.Timestamp("1976-01-08")]
ev = ev[ev.cls.isin(["INT", "OR", "CRU", "other"]) & (ev.depth <= 60)]
rows = []
for _, e in ev.iterrows():
    fb = F[F.bin_id == e.bin_id]
    prev = fb[fb.t_start <= e.time].sort_values("t_start").iloc[-1]
    r = {"time": str(e.time)[:10], "mag": e.mag, "lat": e.latitude, "depth": e.depth, "cls": e.cls,
         "split": "test" if e.time >= H.TEST_START else "train"}
    for c in ["x_ids_anom", "x_or_anom", "x_vert_shift"]:
        r[c] = round(float(prev[c]), 3)
        r[c + "_pct"] = round(float((fb[c] < prev[c]).mean() * 100), 1)
    rows.append(r)
d = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(d.to_string())
for sp in ("train", "test", None):
    s = d if sp is None else d[d.split == sp]
    print(sp or "all", "n=", len(s), "median percentile:",
          {c: float(s[c + "_pct"].median()) for c in ["x_ids_anom", "x_or_anom", "x_vert_shift"]},
          "frac>80th pct x_ids:", round(float((s.x_ids_anom_pct > 80).mean()), 3))
d.to_csv(f"{out}/event_level_m65.csv", index=False)
