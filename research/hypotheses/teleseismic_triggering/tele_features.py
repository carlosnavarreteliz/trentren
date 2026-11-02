"""Causal teleseismic / global-synchrony features aligned to the harness panel.

Global catalog: USGS ComCat M>=6.5 (CSV files passed as paths). Remote = > dist_km from a bin's reference point.
All features at issue time t use only events with time < t - 1 hour.
"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np
import pandas as pd
import harness as H

T0 = pd.Timestamp("1973-01-01")
T1 = pd.Timestamp("2026-10-17")
SPAN_DAYS = (T1 - T0).total_seconds() / 86400.0
CUTOFF_DAYS = 1.0 / 24.0          # event must precede t_start by > 1 hour (real-time availability)
FLOOR_MAX = -3.0                  # log10 strain floor when no remote event in window
EPS_FLOOR_SUM = 1e-3


def load_global(paths):
    df = pd.concat([pd.read_csv(p, usecols=["time", "latitude", "longitude", "depth", "mag", "type", "id"])
                    for p in paths], ignore_index=True)
    df = df[df["type"] == "earthquake"].drop_duplicates("id")
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    df = df[(df.time >= T0) & (df.time < T1 + pd.Timedelta(days=1))]
    df = df.dropna(subset=["latitude", "longitude", "mag"]).sort_values("time").reset_index(drop=True)
    df["depth"] = df["depth"].fillna(10.0)
    return df


def bin_reference_points():
    """(lat centre, median lon of pre-2013 M>=4.7 events in the bin) -- static geometry, training data only."""
    ev = H.get_events(min_mag=4.7, end="2013-01-01")
    lon = ev.groupby("bin_id").longitude.median().reindex(range(29)).fillna(-71.0).to_numpy()
    return H.bin_centers(), lon


def gc_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


class TeleFeatures:
    def __init__(self, glob: pd.DataFrame, dist_km: float = 1000.0, max_depth_strain: float = 100.0):
        self.g = glob.reset_index(drop=True)
        blat, blon = bin_reference_points()
        self.blat, self.blon = blat, blon
        D = gc_deg(self.g.latitude.to_numpy()[:, None], self.g.longitude.to_numpy()[:, None],
                   blat[None, :], blon[None, :])                      # [E, B] degrees
        self.D = D
        km = D * 111.195
        self.remote_bin = km > dist_km                                 # [E, B]
        self.remote_all = self.remote_bin.all(axis=1)                  # remote from every bin
        M = self.g.mag.to_numpy()
        shallow = self.g.depth.to_numpy() <= max_depth_strain
        loge = M[:, None] - 1.66 * np.log10(np.maximum(D, 0.1)) - 4.0
        ok = self.remote_bin & shallow[:, None]
        self.loge = np.where(ok, loge, -np.inf)                        # [E, B] log10 strain proxy (micro-strain-ish)
        self.eps = np.where(ok, 10.0 ** loge, 0.0)
        self.M0 = 10 ** (1.5 * M + 9.1)
        lon, lat = self.g.longitude.to_numpy(), self.g.latitude.to_numpy()
        self.pacific = ((lon >= 100) | (lon <= -60)) & (np.abs(lat) <= 65)
        self.mag = M
        self.t_days = ((self.g.time - T0).dt.total_seconds() / 86400.0).to_numpy()

    # ---- shifted event times (surrogate) -------------------------------------------------------
    def times(self, shift_days=0.0):
        if not shift_days:
            return self.t_days, np.arange(len(self.t_days))
        t = np.mod(self.t_days + shift_days, SPAN_DAYS)
        order = np.argsort(t, kind="stable")
        return t[order], order

    # ---- per-bin H1 features --------------------------------------------------------------------
    def h1(self, panel, shift_days=0.0, windows_max=(7,), windows_sum=(30,)):
        t, order = self.times(shift_days)
        loge, eps = self.loge[order], self.eps[order]
        ts = np.sort(panel.t_start.unique())
        ti = ((pd.DatetimeIndex(ts) - T0).total_seconds() / 86400.0).to_numpy()
        hi = np.searchsorted(t, ti - CUTOFF_DAYS, side="left")       # events with t < ti - cutoff
        cs = np.vstack([np.zeros((1, eps.shape[1])), np.cumsum(eps, axis=0)])
        B = eps.shape[1]
        out = {}
        for w in windows_max:
            lo = np.searchsorted(t, ti - CUTOFF_DAYS - w, side="left")
            m = np.full((len(ti), B), -np.inf)
            nz = np.where(hi > lo)[0]
            for i in nz:
                m[i] = loge[lo[i]:hi[i]].max(axis=0)
            out[f"x_strain_max_{w}d"] = np.maximum(m, FLOOR_MAX)
        for w in windows_sum:
            lo = np.searchsorted(t, ti - CUTOFF_DAYS - w, side="left")
            s = cs[hi] - cs[lo]
            out[f"x_strain_sum_{w}d"] = np.log10(EPS_FLOOR_SUM + np.maximum(s, 0))
        df = pd.DataFrame({"t_start": np.repeat(ts, B), "bin_id": np.tile(np.arange(B), len(ts))})
        for k, v in out.items():
            df[k] = v.ravel()
        return df

    # ---- global H2 features ---------------------------------------------------------------------
    def h2(self, panel, shift_days=0.0, diff=False):
        t, order = self.times(shift_days)
        rem = self.remote_all[order]
        M0 = np.where(rem, self.M0[order], 0.0)
        m7 = (rem & self.pacific[order] & (self.mag[order] >= 7.0)).astype(float)
        ts = np.sort(panel.t_start.unique())
        ti = ((pd.DatetimeIndex(ts) - T0).total_seconds() / 86400.0).to_numpy()
        hi = np.searchsorted(t, ti - CUTOFF_DAYS, side="left")
        lo = np.searchsorted(t, ti - CUTOFF_DAYS - 365.25, side="left")
        cM, c7 = np.r_[0, np.cumsum(M0)], np.r_[0, np.cumsum(m7)]
        lm = np.log10(1e18 + cM[hi] - cM[lo])
        s = pd.Series(lm, index=pd.DatetimeIndex(ts))
        trail = s.rolling("3652D", closed="left", min_periods=1).mean()   # strictly earlier issue times
        anom = (s - trail).to_numpy()
        anom[0] = 0.0
        p7 = c7[hi] - c7[lo]
        df = pd.DataFrame({"t_start": ts, "x_gmoment_365d": anom, "x_pacific_m7_365d": p7})
        if diff:   # causal first difference over 52 weeks-equivalent (365 d) -- detrended variant
            lag = max(1, int(round(365.25 / max(1, (ts[1] - ts[0]) / np.timedelta64(1, "D")))))
            for c in ["x_gmoment_365d", "x_pacific_m7_365d"]:
                v = df[c].to_numpy()
                d = np.full_like(v, np.nan, dtype=float)
                d[lag:] = v[lag:] - v[:-lag]
                df[c] = d
        return df
