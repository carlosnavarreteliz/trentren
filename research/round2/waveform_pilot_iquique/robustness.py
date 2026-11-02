"""Robustness / leakage checks (written after the preregistered analysis; all exploratory, disclosed as such).

Usage: /opt/miniconda3/bin/python3 -I robustness.py <picks_dir> <assoc_dir> <meas_dir> <catalog_csv> <out_dir>
R1  'Beyond the routine catalog': S1 PRE after masking enhanced events within T_MASK days after (and 1 day before)
    any ComCat M>=4 event within 100 km (removes the micro-aftershocks/foreshocks of catalogued events).
R2  Timing: first 10-day block (PRE) where the enhanced rate exceeds the control maximum vs the first block with a
    ComCat M>=4 event in R; i.e. did the micro-catalog see anything BEFORE the routine catalog?
R3  Routine-catalog base rate: fraction of 85-day windows 2001-2013 (before target) whose ComCat M>=4 count in R
    is >= the target PRE count (how unusual is a swarm like Jan-Mar 2014, using ~13 yr instead of 2 controls).
R4  Detection homogeneity: daily P-pick counts per station (median) per window; magnitude offset per window.
R5  Magnitude-calibration sensitivity of b: b with each window's own ComCat offset and Mc per window.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

pdir, adir, mdir, catf, odir = map(Path, sys.argv[1:6])
WINS = {"T2014": 2014, "C2013": 2013, "C2017": 2017}
REF = (-19.610, -70.769)
def epi(lat, lon, la, lo):
    dx = (lon - lo) * 111.19 * np.cos(np.radians(0.5 * (lat + la))); dy = (lat - la) * 111.19
    return np.sqrt(dx ** 2 + dy ** 2)
def bounds(y):
    return pd.Timestamp(f"{y-1}-12-22"), pd.Timestamp(f"{y}-03-16 21:00"), pd.Timestamp(f"{y}-04-01 23:46")

res0 = json.load(open(odir / "results.json"))
MC, OFF = res0["Mc_common"], res0["offset_used"]
cat = pd.read_csv(catf); cat["time"] = pd.to_datetime(cat.time, format="mixed").dt.tz_localize(None)
cat = cat[cat.type == "earthquake"]
inR = lambda d: d.latitude.between(-21.0, -18.5) & d.longitude.between(-71.6, -69.8) & (d.depth <= 60)
c4 = cat[(cat.mag >= 4.0) & cat.latitude.between(-22.5, -17.5) & cat.longitude.between(-72.5, -68.0)]

E = {}
for w in WINS:
    ev = pd.read_parquet(adir / f"events_{w}.parquet"); ev = ev[ev.n_sta >= 3].copy()
    am = pd.read_parquet(mdir / f"amps_{w}.parquet"); am = am[am.A_mm > 0]
    am["ml"] = np.log10(am.A_mm) + 1.11 * np.log10(am.R_km) + 0.00189 * am.R_km + 0.591
    ev = ev.merge(am.groupby("event_id").ml.median().rename("ml_raw"), left_on="event_id", right_index=True, how="left")
    ev["mag"] = ev.ml_raw + OFF
    E[w] = ev[inR(ev)].copy()

out = {}
# R1
for TM in (2.0, 7.0):
    r = {}
    for w, y in WINS.items():
        t0, tp, _ = bounds(y)
        e = E[w][(E[w].mag >= MC) & (E[w].origin >= t0) & (E[w].origin < tp)]
        cc = c4[(c4.time >= t0 - pd.Timedelta(days=TM)) & (c4.time < tp)]
        grid = pd.date_range(t0, tp, freq="h")
        masked = np.zeros(len(grid), bool)
        keep = np.ones(len(e), bool)
        for _, c in cc.iterrows():
            a, b = c.time - pd.Timedelta(days=1), c.time + pd.Timedelta(days=TM)
            masked |= (grid >= a) & (grid < b)
            near = epi(e.latitude.values, e.longitude.values, c.latitude, c.longitude) <= 100
            keep &= ~(near & (e.origin >= a).values & (e.origin < b).values)
        # conservative: drop masked time entirely (any location) for the denominator and numerator
        hr = e.origin.dt.floor("h")
        keep_t = ~pd.Series(hr.values).isin(grid[masked]).values
        nd = (~masked).sum() / 24
        r[w] = dict(n_comcat_M4=int(len(cc)), unmasked_days=float(nd),
                    rate_time_masked=float(keep_t.sum() / nd) if nd > 0 else None,
                    rate_space_time_masked=float(keep.sum() / ((tp - t0).total_seconds() / 86400)))
    out[f"R1_mask_{TM:g}d"] = r
# R2
blocks = {}
for w, y in WINS.items():
    t0, tp, _ = bounds(y); rows = []
    b0 = t0
    while b0 + pd.Timedelta(days=10) <= tp:
        b1 = b0 + pd.Timedelta(days=10)
        e = E[w][(E[w].mag >= MC) & (E[w].origin >= b0) & (E[w].origin < b1)]
        cc = c4[(c4.time >= b0) & (c4.time < b1) & inR(c4)]
        rows.append(dict(start=str(b0.date()), enh_rate=len(e) / 10, comcat_M4=int(len(cc)),
                         comcat_maxM=float(cc.mag.max()) if len(cc) else None,
                         med_dist_km=float(np.median(epi(e.latitude.values, e.longitude.values, *REF))) if len(e) else None))
        b0 = b1
    blocks[w] = rows
cmax = max(r["enh_rate"] for w in ("C2013", "C2017") for r in blocks[w])
cmax_cc = max(r["comcat_M4"] for w in ("C2013", "C2017") for r in blocks[w])
out["R2_blocks"] = blocks
out["R2_first_enh_exceed"] = next((r["start"] for r in blocks["T2014"] if r["enh_rate"] > cmax), None)
out["R2_first_comcat_exceed"] = next((r["start"] for r in blocks["T2014"] if r["comcat_M4"] > cmax_cc), None)
out["R2_control_max"] = dict(enh=cmax, comcat_M4=cmax_cc)
# R3
cR = c4[inR(c4)].sort_values("time")
t0, tp, _ = bounds(2014)
nT = int(((cR.time >= t0) & (cR.time < tp)).sum())
L = tp - t0
starts = pd.date_range("2001-01-01", "2013-12-21", freq="7D")
cnt = np.array([((cR.time >= s) & (cR.time < s + L)).sum() for s in starts])
out["R3_comcat"] = dict(target_PRE_count_M4=nT, n_windows=int(len(cnt)), frac_ge_target=float((cnt >= nT).mean()),
                        max_count=int(cnt.max()), median=float(np.median(cnt)),
                        windows_ge_target=sorted({str(s.date())[:7] for s, c in zip(starts, cnt) if c >= nT}))
for y in (2008, 2010, 2011, 2012, 2013):
    pass
# R4
r4 = {}
for w in WINS:
    u = pd.concat([pd.read_csv(f) for f in pdir.glob(f"uptime_{w}_*.csv")])
    r4[w] = u.groupby("station")[["n_P", "n_S"]].median().to_dict()
    r4[w]["comcat_offset"] = res0["calibration"][w]["offset"]
out["R4_detection"] = r4
# R5
from math import log10, e as EE
def bval(m, mc):
    m = m[m >= mc - 0.05]
    return (float(log10(EE) / (m.mean() - (mc - 0.05))), int(len(m))) if len(m) >= 30 else (None, int(len(m)))
r5 = {}
for w, y in WINS.items():
    t0, tp, _ = bounds(y)
    ev = E[w][(E[w].origin >= t0) & (E[w].origin < tp)]
    own = ev.ml_raw + res0["calibration"][w]["offset"]
    r5[w] = {f"Mc{mc}": bval(own.dropna(), mc) for mc in (1.7, 2.0, 2.3)}
out["R5_b_own_offset"] = r5
json.dump(out, open(odir / "robustness.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "R2_blocks"}, indent=1, default=str))
for w in WINS:
    print(w, [(r["start"][5:], r["enh_rate"], r["comcat_M4"], None if r["med_dist_km"] is None else round(r["med_dist_km"])) for r in blocks[w]])
