"""Post-hoc (descriptive): surprise components in the epicentral bin at the last issue time before each test-period
M>=7 event, as percentiles of the training-period distribution. Usage: python3 -I diagnose.py <data_dir>"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np, pandas as pd
import harness as H

dd = sys.argv[1]
F = pd.read_parquet(f"{dd}/features_m4.7_s7_h7.parquet")
panel = H.get_panel()
D = F.merge(panel[["bin_id", "t_start", "split"]], on=["bin_id", "t_start"])
tr = D[D.split == "train"]
cols = ["x_surprise", "x_rr90", "x_ad", "x_gap", "x_lld365", "x_mag", "x_spat", "x_depth"]
ev = H.get_events(min_mag=7.0, start="2013-01-01")
rows = []
for _, e in ev.iterrows():
    ts = D[(D.bin_id == e.bin_id) & (D.t_start <= e.time)].t_start.max()
    r = D[(D.bin_id == e.bin_id) & (D.t_start == ts)].iloc[0]
    row = {"date": str(e.time.date()), "M": e.mag, "lat": round(e.latitude, 2), "issue": str(ts.date())}
    for c in cols:
        ref = tr[c].dropna().to_numpy()
        row[c] = np.nan if pd.isna(r[c]) else round(100 * np.mean(ref <= r[c]), 1)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
