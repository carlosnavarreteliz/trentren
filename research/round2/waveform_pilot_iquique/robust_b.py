"""R6: b-value of target PRE after masking micro-events near routine-catalog M>=4 events; R7: ML-proxy vs ComCat slope.
Usage: python3 -I robust_b.py <assoc_dir> <meas_dir> <catalog_csv> <out_dir>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
adir, mdir, catf, odir = map(Path, sys.argv[1:5])
res0 = json.load(open(odir / "results.json")); MC, OFF = res0["Mc_common"], res0["offset_used"]
def epi(lat, lon, la, lo):
    return np.hypot((lon - lo) * 111.19 * np.cos(np.radians(0.5 * (lat + la))), (lat - la) * 111.19)
cat = pd.read_csv(catf); cat["time"] = pd.to_datetime(cat.time, format="mixed").dt.tz_localize(None)
cat = cat[(cat.type == "earthquake") & cat.latitude.between(-22.5, -17.5) & cat.longitude.between(-72.5, -68.0)]
inR = lambda d: d.latitude.between(-21.0, -18.5) & d.longitude.between(-71.6, -69.8) & (d.depth <= 60)
out = {}
allm = []
for w, y in {"T2014": 2014, "C2013": 2013, "C2017": 2017}.items():
    t0, tp = pd.Timestamp(f"{y-1}-12-22"), pd.Timestamp(f"{y}-03-16 21:00")
    ev = pd.read_parquet(adir / f"events_{w}.parquet"); ev = ev[ev.n_sta >= 3]
    am = pd.read_parquet(mdir / f"amps_{w}.parquet"); am = am[am.A_mm > 0]
    am["ml"] = np.log10(am.A_mm) + 1.11 * np.log10(am.R_km) + 0.00189 * am.R_km + 0.591
    ev = ev.merge(am.groupby("event_id").ml.median().rename("ml_raw"), left_on="event_id", right_index=True)
    ev["mag"] = ev.ml_raw + OFF
    # matches for slope
    cw = cat[(cat.time >= t0) & (cat.time < pd.Timestamp(f"{y}-04-01 23:46"))]
    for _, c in cw.iterrows():
        dt = (ev.origin - c.time).dt.total_seconds().abs()
        cand = ev[dt <= 10]
        if len(cand):
            d = epi(cand.latitude.values, cand.longitude.values, c.latitude, c.longitude)
            if (d <= 120).any():
                allm.append((w, c.mag, cand.ml_raw.values[np.argmin(np.where(d <= 120, dt[cand.index], 1e9))]))
    e = ev[inR(ev) & (ev.origin >= t0) & (ev.origin < tp)]
    c4 = cat[(cat.mag >= 4) & (cat.time >= t0 - pd.Timedelta(days=7)) & (cat.time < tp)]
    r = {}
    for TM in (0, 2, 7):
        keep = np.ones(len(e), bool)
        if TM:
            for _, c in c4.iterrows():
                near = epi(e.latitude.values, e.longitude.values, c.latitude, c.longitude) <= 100
                keep &= ~(near & (e.origin >= c.time - pd.Timedelta(days=1)).values & (e.origin < c.time + pd.Timedelta(days=TM)).values)
        m = e.mag[keep]; m = m[m >= MC - 0.05]
        r[f"mask{TM}d"] = (float(np.log10(np.e) / (m.mean() - (MC - 0.05))) if len(m) >= 30 else None, int(len(m)))
    out[w] = r
A = pd.DataFrame(allm, columns=["w", "mag", "ml"])
sl = {w: (float(np.polyfit(g.ml, g.mag, 1)[0]), len(g)) for w, g in A.groupby("w")}
sl["all"] = (float(np.polyfit(A.ml, A.mag, 1)[0]), len(A))
out["slope_comcat_on_proxy"] = sl
json.dump(out, open(odir / "robust_b.json", "w"), indent=1)
print(json.dumps(out, indent=1))
