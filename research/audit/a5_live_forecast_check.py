"""A5: sanity-check the live zone forecast (results/live_chile_forecast_oct2026.csv) against plain
climatology: empirical probability that a 30-day window contains >=1 event of M>=X with epicenter in the zone."""
import json
import sys
import numpy as np
import pandas as pd
ROOT = "/Users/cnavarreteliz/Documents/Repos/earthquakes"
raw = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/data/raw/chile_usgs_1900_2026.csv"
cat = pd.read_csv(raw, usecols=["time", "latitude", "longitude", "mag"])
cat["time"] = pd.to_datetime(cat["time"], utc=True, format="ISO8601")
live = pd.read_csv(f"{ROOT}/results/live_chile_forecast_oct2026.csv")
feat_cols = pd.read_csv(f"{ROOT}/data/processed/chile_features_m40.csv", nrows=1).columns
zones = [(-19.5, -17.0), (-21.5, -19.5), (-25.0, -21.5), (-29.0, -25.0), (-32.2, -29.0), (-34.5, -32.2),
         (-38.2, -34.5), (-46.5, -38.2)]

def window_prob(sub, t0, t1, H=30):
    starts = pd.date_range(t0, t1 - pd.Timedelta(days=H), freq="D", tz="UTC")
    ev = np.sort(sub["time"].values.astype("datetime64[s]").astype(np.int64))
    s = starts.values.astype("datetime64[s]").astype(np.int64)
    i0 = np.searchsorted(ev, s, "left")
    i1 = np.searchsorted(ev, s + H * 86400, "left")
    return float(np.mean(i1 > i0))

rows = []
for (lo, hi), (_, r) in zip(zones, live.iterrows()):
    z = cat[(cat["latitude"] >= lo) & (cat["latitude"] <= hi) & (cat["longitude"] <= -66.0)]
    row = {"zone": r["zone"]}
    for M, col in [(5.0, "prob_m50_30d"), (5.5, "prob_m55_30d"), (6.0, "prob_m60_30d")]:
        zz = z[z["mag"] >= M]
        row[f"live_{col}"] = float(r[col])
        row[f"clim_1976_2012_M{M}"] = window_prob(zz, pd.Timestamp("1976-01-01", tz="UTC"), pd.Timestamp("2013-01-01", tz="UTC"))
        row[f"clim_2018_2026_M{M}"] = window_prob(zz, pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC"))
    rows.append(row)
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
out = {"zones": res.round(4).to_dict(orient="records"),
       "bouchon_or_deep_to_shallow_column_exists": bool({"bouchon_slab_pull_ratio", "deep_to_shallow_90d"} & set(feat_cols)),
       "b_pos_100_at_clip_ceiling_1.28": int(np.sum(np.isclose(live["b_pos_100"], 1.28)))}
print(out["bouchon_or_deep_to_shallow_column_exists"], out["b_pos_100_at_clip_ceiling_1.28"])
json.dump(out, open(f"{ROOT}/research/audit/a5_results.json", "w"), indent=1)
