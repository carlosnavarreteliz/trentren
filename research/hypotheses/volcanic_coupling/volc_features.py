"""Causal per-bin volcanic-arc features for the harness panel (volcanic_coupling hypothesis).

load_sources(data_dir) -> (volcanoes, eruptions, arc_events)
build_features(panel, eruptions, arc_events, shift_days=0) -> DataFrame[bin_id, t_start, x_arc_365, x_arc_30, x_erupt_365]
"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np
import pandas as pd
import harness as H

NB = 29          # panel bins (1 deg, -46..-17)
LAT0 = -46.0


def _read_gvp(path):
    d = json.load(open(path))
    return pd.DataFrame([dict(f["properties"], lon=f["geometry"]["coordinates"][0], lat=f["geometry"]["coordinates"][1])
                         for f in d["features"] if f.get("geometry")])


def load_sources(data_dir, arc_mmin=4.5, arc_depth=40.0, arc_dist_km=50.0):
    v = _read_gvp(f"{data_dir}/Smithsonian_VOTW_Holocene_Volcanoes.json")
    v = v[(v.lat > -48) & (v.lat < -15) & (v.lon > -76) & (v.lon < -62)].reset_index(drop=True)
    e = _read_gvp(f"{data_dir}/Smithsonian_VOTW_Holocene_Eruptions.json")
    e = e[(e.lat > -48) & (e.lat < -15) & (e.lon > -76) & (e.lon < -62)]
    e = e[(e.Activity_Type == "Confirmed Eruption") & e.StartDateYear.notna() & e.StartDateMonth.notna()
          & e.StartDateDay.notna() & (e.StartDateYear >= 1960)].copy()
    unc = e.StartDateDayUncertainty.fillna(0)
    e = e[unc <= 31].copy()
    e["onset"] = pd.to_datetime(dict(year=e.StartDateYear.astype(int), month=e.StartDateMonth.astype(int),
                                     day=e.StartDateDay.astype(int)), errors="coerce")
    e = e[e.onset.notna()]
    # conservative "known-by" time: onset + uncertainty + 1 day reporting lag
    e["known"] = e.onset + pd.to_timedelta(e.StartDateDayUncertainty.fillna(0) + 1, unit="D")
    e = e.sort_values("known").reset_index(drop=True)

    cat = H.load_catalog(min_mag=None)
    lat = np.radians(cat.latitude.to_numpy())[:, None]
    vl = np.radians(v.lat.to_numpy())[None, :]
    dn = np.radians(v.lon.to_numpy())[None, :] - np.radians(cat.longitude.to_numpy())[:, None]
    a = np.sin((vl - lat) / 2) ** 2 + np.cos(lat) * np.cos(vl) * np.sin(dn / 2) ** 2
    cat["dvol_km"] = (2 * 6371 * np.arcsin(np.sqrt(a))).min(1)
    arc = cat[(cat.depth < arc_depth) & (cat.dvol_km <= arc_dist_km) & (cat.mag >= arc_mmin)].copy()
    arc = arc.sort_values("time").reset_index(drop=True)
    return v, e, arc


def _bin_ext(lat):
    """bin index allowing -1 and NB (just outside the panel) so edge bins see their outside neighbours."""
    return np.floor((np.asarray(lat, float) - LAT0)).astype(int)


def _count_nbr(times, lats, issue, window_days, nbr=1):
    """counts[t, b] = # points with bin in [b-nbr, b+nbr] and issue[t]-window <= time < issue[t]."""
    tb = _bin_ext(lats)
    tt = np.asarray(times, dtype="datetime64[ns]")
    out = np.zeros((len(issue), NB))
    iss = np.asarray(issue, dtype="datetime64[ns]")
    lo = iss - np.timedelta64(int(window_days * 86400), "s")
    for b in range(NB):
        m = (tb >= b - nbr) & (tb <= b + nbr)
        ts = np.sort(tt[m])
        out[:, b] = np.searchsorted(ts, iss, side="left") - np.searchsorted(ts, lo, side="left")
    return out


def _shift_times(times, shift_days, t0=np.datetime64("1960-01-01"), t1=np.datetime64("2026-10-17")):
    """Circularly shift point times within [t0, t1)."""
    if not shift_days:
        return np.asarray(times, dtype="datetime64[ns]")
    span = (t1 - t0).astype("timedelta64[s]").astype(np.int64)
    s = (np.asarray(times, dtype="datetime64[s]") - t0).astype(np.int64)
    s = (s + int(shift_days) * 86400) % span
    return (t0 + s.astype("timedelta64[s]")).astype("datetime64[ns]")


def build_features(panel, eruptions, arc, shift_days=0, cols=("x_arc_365", "x_arc_30", "x_erupt_365")):
    issue = np.sort(panel.t_start.unique())
    at = _shift_times(arc.time.to_numpy(), shift_days)
    et = _shift_times(eruptions.known.to_numpy(), shift_days)
    feats = {
        "x_arc_365": np.log1p(_count_nbr(at, arc.latitude, issue, 365)),
        "x_arc_30": np.log1p(_count_nbr(at, arc.latitude, issue, 30)),
        "x_erupt_365": _count_nbr(et, eruptions.lat, issue, 365),
    }
    T = len(issue)
    df = pd.DataFrame({"t_start": np.repeat(issue, NB), "bin_id": np.tile(np.arange(NB), T)})
    for c in cols:
        df[c] = feats[c].reshape(-1)
    df["bin_id"] = df["bin_id"].astype(panel.bin_id.dtype)
    return df
