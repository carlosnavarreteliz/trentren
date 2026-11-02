"""Associate PhaseNetLight picks into events with PyOcto (homogeneous model), per window.

Usage: /opt/miniconda3/bin/python3 -I associate.py <picks_dir> <out_dir>
Output: <out_dir>/events_<win>.parquet (origin time, lat, lon, depth, n picks, n stations),
        <out_dir>/assign_<win>.parquet (pick-event assignments)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyocto

pdir, odir = Path(sys.argv[1]), Path(sys.argv[2]); odir.mkdir(parents=True, exist_ok=True)
STATIONS = pd.DataFrame({
    "id": ["PSGCX", "PB11", "PATCX"],
    "latitude": [-19.59717, -19.76096, -20.82071],
    "longitude": [-70.12305, -69.65582, -70.15288],
    "elevation": [966.0, 1400.0, 832.0],
})
vel = pyocto.VelocityModel0D(p_velocity=6.8, s_velocity=3.9, tolerance=1.5)
assoc = pyocto.OctoAssociator.from_area(
    lat=(-22.5, -17.5), lon=(-72.5, -68.0), zlim=(0, 200), time_before=60.0, velocity_model=vel,
    n_picks=4, n_p_picks=2, n_s_picks=1, n_p_and_s_picks=1, min_pick_fraction=0.25, n_threads=2)
stations = assoc.transform_stations(STATIONS.copy())

for win in ["T2014", "C2013", "C2017"]:
    fs = sorted(pdir.glob(f"picks_{win}_*.parquet"))
    if len(fs) < 3:
        print("missing picks for", win, fs); continue
    picks = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    picks["time"] = pd.to_datetime(picks["time"]).astype("int64") / 1e9
    picks = picks.sort_values("time").reset_index(drop=True)
    picks["peak_value"] = picks["prob"]
    print(win, "picks", len(picks), picks.groupby(["station", "phase"]).size().to_dict(), flush=True)
    ev_all, as_all = [], []
    # process in daily chunks (plus 5 min overlap) to bound memory; dedupe by event id ownership of start day
    t0, t1 = picks.time.min(), picks.time.max()
    day = 86400.0
    start = np.floor(t0 / day) * day
    while start < t1:
        sel = picks[(picks.time >= start - 300) & (picks.time < start + day + 300)].copy()
        if len(sel) >= 4:
            ev, asg = assoc.associate(sel.reset_index(drop=True), stations)
            if len(ev):
                ev = assoc.transform_events(ev)
                keep = (ev.time >= start) & (ev.time < start + day)
                ev = ev[keep]
                asg = asg[asg.event_idx.isin(ev.idx)]
                sel2 = sel.reset_index(drop=True)
                asg = asg.merge(sel2[["station", "phase", "time", "prob"]], left_on="pick_idx", right_index=True,
                                suffixes=("", "_pick"))
                key = f"{int(start)}_"
                ev = ev.assign(event_id=key + ev.idx.astype(str))
                asg = asg.assign(event_id=key + asg.event_idx.astype(str))
                ev_all.append(ev); as_all.append(asg)
        start += day
    ev = pd.concat(ev_all, ignore_index=True)
    asg = pd.concat(as_all, ignore_index=True)
    nst = asg.groupby("event_id").station.nunique().rename("n_sta")
    ev = ev.merge(nst, left_on="event_id", right_index=True, how="left")
    ev["origin"] = pd.to_datetime(ev.time, unit="s")
    ev["window"] = win
    print(win, "events", len(ev), ">=3 sta", int((ev.n_sta >= 3).sum()), flush=True)
    ev.to_parquet(odir / f"events_{win}.parquet"); asg.to_parquet(odir / f"assign_{win}.parquet")
