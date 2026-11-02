"""Feature construction for the vertical_coupling hypothesis (see PREREGISTRATION.md).

All features at issue time t use only events with time < t (and GCMT mechanisms with time < t - 30 d).
Geometry: Slab1.0 (Hayes et al. 2012, pre-2012 data) + USGS 2017 trench trace (static bathymetry).
"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import glob
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from scipy.interpolate import RegularGridInterpolator
import harness as H

DATA = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/vertical_coupling"
NB = 29


# ---------------------------------------------------------------- geometry
def load_geometry(data_dir=DATA):
    f = netcdf_file(f"{data_dir}/sam_slab1.0_augmented.grd", "r", mmap=False)
    x = f.variables["x"][:].astype(float)
    y = f.variables["y"][:].astype(float)
    z = -f.variables["z"][:].astype(float)  # positive depth km
    f.close()
    interp = RegularGridInterpolator((y, x), z, bounds_error=False, fill_value=np.nan)
    tr = pd.read_csv(f"{data_dir}/trenches_usgs_2017.csv")
    tr = tr[(tr.slab == "sam") & (tr.lat < -14) & (tr.lat > -48)].sort_values("lat")
    tr_lat = tr.lat.to_numpy(float)
    tr_lon = np.where(tr.lon.to_numpy(float) > 180, tr.lon.to_numpy(float) - 360, tr.lon.to_numpy(float))
    return interp, tr_lat, tr_lon


def slab_depth(interp, lat, lon):
    lat = np.asarray(lat, float)
    lon360 = np.mod(np.asarray(lon, float), 360)
    latc = np.clip(lat, -45.0, -3.0)  # grid starts at -45: use nearest row for -46..-45
    return interp(np.column_stack([latc, lon360]))


def classify(ev, geom=None):
    interp, tr_lat, tr_lon = geom or load_geometry()
    lat, lon, z = ev.latitude.to_numpy(float), ev.longitude.to_numpy(float), ev.depth.to_numpy(float)
    lon_tr = np.interp(lat, tr_lat, tr_lon)
    zs = slab_depth(interp, lat, lon)
    seaward = lon < lon_tr
    dland = lon - lon_tr  # deg landward of trench
    cls = np.full(len(ev), "other", dtype=object)
    cls[seaward & (z <= 60)] = "OR"
    land = ~seaward
    cls[land & (z >= 70) & (z <= 300)] = "IDS"
    intf = land & (z <= 60) & ((np.isfinite(zs) & (np.abs(z - zs) <= 20)) | ((dland <= 0.3) & (z <= 40)))
    cls[intf] = "INT"
    cru = land & (z <= 40) & np.isfinite(zs) & (z < zs - 20) & ~intf
    cls[cru] = "CRU"
    out = ev.copy()
    out["cls"] = cls
    out["slab_z"] = zs
    out["d_trench_deg"] = dland
    return out


# ---------------------------------------------------------------- counting
def window_counts(ev, ts_days, lo, hi, nbr=1):
    """count events in [t - lo, t - hi) for bins within +-nbr. returns [T, NB]."""
    eb = ev.bin_id.to_numpy()
    et = ev.t_days.to_numpy(float)
    out = np.zeros((len(ts_days), NB))
    for b in range(NB):
        tb = np.sort(et[(np.abs(eb - b) <= nbr) & (eb >= 0)])
        out[:, b] = np.searchsorted(tb, ts_days - hi, side="left") - np.searchsorted(tb, ts_days - lo, side="left")
    return out


T0 = float(H.to_days(pd.Timestamp("1973-01-01")))


def rate_anom(ev, ts_days, short, long_, nbr=1, min_long=365):
    ns = window_counts(ev, ts_days, short, 0, nbr)
    # long window [t-short-long, t-short), truncated at catalog start
    avail = np.minimum(long_, ts_days - short - T0)
    nl = window_counts(ev, ts_days, short + long_, short, nbr)
    L = avail[:, None] * np.ones((1, NB))
    exp_ = np.where(L > 0, nl * short / np.maximum(L, 1), np.nan)
    x = np.log((ns + 0.5) / (exp_ + 0.5))
    x[L < min_long] = np.nan
    return x


def vert_shift(ids, intf, ts_days, short, long_, nbr=1, min_long=365):
    a_s = window_counts(ids, ts_days, short, 0, nbr)
    b_s = window_counts(intf, ts_days, short, 0, nbr)
    a_l = window_counts(ids, ts_days, short + long_, short, nbr)
    b_l = window_counts(intf, ts_days, short + long_, short, nbr)
    fs = (a_s + 0.5) / (a_s + b_s + 1)
    fl = (a_l + 0.5) / (a_l + b_l + 1)
    x = np.log(fs / (1 - fs)) - np.log(fl / (1 - fl))
    avail = np.minimum(long_, ts_days - short - T0)
    x[(avail < min_long)] = np.nan
    return x


def to_frame(ts, cols: dict):
    T = len(ts)
    tt, bb = np.meshgrid(np.arange(T), np.arange(NB), indexing="ij")
    d = {"bin_id": bb.ravel(), "t_start": pd.DatetimeIndex(ts)[tt.ravel()]}
    for k, v in cols.items():
        d[k] = v.ravel()
    return pd.DataFrame(d)


def build_F1(panel, ev_cls, short_ids=180, bg_only=False):
    ts = pd.DatetimeIndex(np.sort(panel.t_start.unique()))
    td = H.to_days(ts)
    e = ev_cls[ev_cls.bg] if bg_only else ev_cls
    ids, orr, intf = e[e.cls == "IDS"], e[e.cls == "OR"], e[e.cls == "INT"]
    cols = {
        "x_ids_anom": rate_anom(ids, td, short_ids, 1825),
        "x_or_anom": rate_anom(orr, td, 365, 3650),
        "x_vert_shift": vert_shift(ids, intf, td, short_ids, 1825),
    }
    return to_frame(ts, cols)


# ---------------------------------------------------------------- GCMT
def parse_ndk(paths):
    rows = []
    for p in paths:
        with open(p, "r", errors="replace") as fh:
            lines = fh.read().splitlines()
        for i in range(0, len(lines) - 4, 5):
            l1, l5 = lines[i], lines[i + 4]
            try:
                date = l1[5:15].strip(); tm = l1[16:26].strip()
                lat = float(l1[27:33]); lon = float(l1[34:41]); dep = float(l1[42:47])
                f5 = l5.split()
                # f5: version, T(val,pl,az), N(val,pl,az), P(val,pl,az), M0, strike, dip, rake, ...
                tpl, ppl = float(f5[2]), float(f5[8])
                mexp = int(lines[i + 3].split()[0])
                m0 = float(f5[10]) * 10 ** mexp
            except Exception:
                continue
            t = pd.to_datetime(date.replace("/", "-") + " " + tm[:8].replace(" ", "0"), errors="coerce")
            rows.append((t, lat, lon, dep, tpl, ppl, m0))
    d = pd.DataFrame(rows, columns=["time", "latitude", "longitude", "depth", "t_plunge", "p_plunge", "m0"])
    d = d.dropna(subset=["time"]).drop_duplicates(subset=["time", "latitude", "longitude"]).sort_values("time")
    d["mw"] = (2 / 3) * (np.log10(d.m0) - 16.1)
    d["mech"] = np.where(d.t_plunge > 50, "thrust", np.where(d.p_plunge > 50, "normal", "other"))
    return d.reset_index(drop=True)


def load_gcmt(data_dir=DATA):
    paths = [f"{data_dir}/jan76_dec20.ndk"] + sorted(glob.glob(f"{data_dir}/monthly/*.ndk")) + [f"{data_dir}/qcmt.ndk"]
    d = parse_ndk(paths)
    d = d[d.latitude.between(-47, -16) & d.longitude.between(-80, -66)].copy()
    d["t_days"] = H.to_days(d.time)
    d["bin_id"] = H.lat_to_bin(d.latitude.to_numpy())
    return d.reset_index(drop=True)


def build_or_comp(panel, gcmt_cls, window=1825, lag=30, nbr=2):
    ts = pd.DatetimeIndex(np.sort(panel.t_start.unique()))
    td = H.to_days(ts)
    o = gcmt_cls[gcmt_cls.cls == "OR"]
    nt = window_counts(o[o.mech == "thrust"], td, window + lag, lag, nbr)
    nn = window_counts(o[o.mech == "normal"], td, window + lag, lag, nbr)
    x = (nt - nn) / (nt + nn + 1)
    return to_frame(ts, {"x_or_comp": x})
