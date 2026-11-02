"""Causal Earth-rotation features (LOD deceleration, pole tide, Chandler amplitude) aligned to the harness panel.

All measured EOP values are used only for days D < t_start - LAG (LAG = 30 d, C04 publication lag).
"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np
import pandas as pd
import harness as H

LAG = 30                 # days: C04 publication lag (file ends 30 d before today)
FIT_DAYS = int(6 * 365.25)
P_ANN, P_CW = 365.25, 433.0
LON_TRENCH = -71.5


def load_eop(path):
    cols = ["yr", "mo", "dy", "hh", "mjd", "x", "y", "ut1", "dX", "dY", "xrt", "yrt", "lod"]
    d = pd.read_csv(path, sep=r"\s+", comment="#", header=None, usecols=range(13), names=cols)
    d["date"] = pd.to_datetime(dict(year=d.yr, month=d.mo, day=d.dy))
    d = d.set_index("date").sort_index()
    return d[["x", "y", "lod"]].astype(float)  # x,y arcsec; lod seconds


# ------------------------------------------------------------------------------------------------ LOD (H1)
def lod_daily_derived(lod_ms: pd.Series) -> pd.DataFrame:
    """Daily causal derived series (value at day D uses days <= D only)."""
    l365 = lod_ms.rolling(365, min_periods=300).mean()
    dlod = l365 - l365.shift(365)
    out = pd.DataFrame({
        "lod_decel_lag5y": dlod.shift(int(round(5 * 365.25))),
        "lod_decel_lag0": dlod,
        "lod365_anom": l365 - l365.rolling(3653, min_periods=2000).mean(),
    })
    return out


def lod_features(lod_ms: pd.Series, panel, cols=("lod_decel_lag5y",)) -> pd.DataFrame:
    der = lod_daily_derived(lod_ms)
    out = None
    for c in cols:
        f = H.align_daily_series(der[c], panel, window_days=7, agg="last", lag_days=LAG, name=f"x_{c}")
        out = f if out is None else out.merge(f, on="t_start")
    return out


# ------------------------------------------------------------------------------------------------ pole tide (H2)
def _issue_times(panel):
    return pd.DatetimeIndex(np.sort(panel.t_start.unique()))


def pole_fit(eop: pd.DataFrame, panel, horizon_days: int):
    """For every issue time t: LS fit of x and y on days D in [t-LAG-FIT_DAYS, t-LAG) with
    const + linear + annual + Chandler terms; evaluate the oscillatory part (and its time derivative) at the
    forecast-window midpoint t + h/2. Returns DataFrame [t_start, m1, m2, m1dot, m2dot, cw_amp] (arcsec, arcsec/yr)."""
    ts = _issue_times(panel)
    d = eop.index.to_numpy()
    day = (eop.index - pd.Timestamp("2000-01-01")).days.to_numpy(float)
    X = eop["x"].to_numpy()
    Y = eop["y"].to_numpy()
    hi = np.searchsorted(d, (ts - pd.Timedelta(days=LAG)).to_numpy(), side="left")
    lo = np.searchsorted(d, (ts - pd.Timedelta(days=LAG + FIT_DAYS)).to_numpy(), side="left")
    wa, wc = 2 * np.pi / P_ANN, 2 * np.pi / P_CW
    res = np.full((len(ts), 5), np.nan)
    for i, t in enumerate(ts):
        if hi[i] - lo[i] < 1000:
            continue
        tm = (t - pd.Timestamp("2000-01-01")).days + horizon_days / 2.0
        dd = day[lo[i]:hi[i]]
        A = np.column_stack([np.ones_like(dd), (dd - tm) / 365.25, np.cos(wa * dd), np.sin(wa * dd),
                             np.cos(wc * dd), np.sin(wc * dd)])
        cx = np.linalg.lstsq(A, X[lo[i]:hi[i]], rcond=None)[0]
        cy = np.linalg.lstsq(A, Y[lo[i]:hi[i]], rcond=None)[0]

        def osc(c):
            v = c[2] * np.cos(wa * tm) + c[3] * np.sin(wa * tm) + c[4] * np.cos(wc * tm) + c[5] * np.sin(wc * tm)
            dv = (-c[2] * wa * np.sin(wa * tm) + c[3] * wa * np.cos(wa * tm)
                  - c[4] * wc * np.sin(wc * tm) + c[5] * wc * np.cos(wc * tm)) * 365.25
            return v, dv
        vx, dvx = osc(cx)
        vy, dvy = osc(cy)
        amp = 0.5 * (np.hypot(cx[4], cx[5]) + np.hypot(cy[4], cy[5]))
        res[i] = [vx, -vy, dvx, -dvy, amp]
    return pd.DataFrame(res, columns=["m1", "m2", "m1dot", "m2dot", "cw_amp"]).assign(t_start=ts)


def pole_tide_panel(pf: pd.DataFrame, panel) -> pd.DataFrame:
    """Per-bin pole-tide radial displacement (mm) and rate (mm/yr), IERS 2010 eq. 7.26."""
    lat = H.bin_centers()
    theta = np.deg2rad(90.0 - lat)
    lam = np.deg2rad(LON_TRENCH)
    g = np.sin(2 * theta)  # per bin
    proj = pf.m1.to_numpy() * np.cos(lam) + pf.m2.to_numpy() * np.sin(lam)
    projd = pf.m1dot.to_numpy() * np.cos(lam) + pf.m2dot.to_numpy() * np.sin(lam)
    T, B = len(pf), len(lat)
    df = pd.DataFrame({
        "t_start": np.repeat(pf.t_start.to_numpy(), B),
        "bin_id": np.tile(np.arange(B), T),
        "x_pt": (-33.0 * proj[:, None] * g[None, :]).ravel(),
        "x_pt_rate": (-33.0 * projd[:, None] * g[None, :]).ravel(),
    })
    return df


def actual_pole_tide_daily(eop: pd.DataFrame, lat=-30.0):
    """Retrospective (non-causal) daily pole tide at a latitude, using the IERS 2018 linear secular pole.
    Only for the exploratory event-level phase test, never as a forecast feature."""
    t = (eop.index - pd.Timestamp("2000-01-01")).days.to_numpy() / 365.25
    xs = (55.0 + 1.677 * t) / 1000.0
    ys = (320.5 + 3.460 * t) / 1000.0
    m1 = eop.x.to_numpy() - xs
    m2 = -(eop.y.to_numpy() - ys)
    lam = np.deg2rad(LON_TRENCH)
    g = np.sin(2 * np.deg2rad(90 - lat))
    s = -33.0 * g * (m1 * np.cos(lam) + m2 * np.sin(lam))
    return pd.Series(s, index=eop.index)
