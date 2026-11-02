"""Causal hydroclimate features aligned to the harness panel. All features at issue time t use days D <= t-2
(i.e. D < t - 1 day, lag_days=1 as in H.align_daily_series). Climatologies fitted on train years only."""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
from pathlib import Path
import numpy as np, pandas as pd
import harness as H

DAY0 = pd.Timestamp("1979-01-01")
TRAIN_END = pd.Timestamp("2013-01-01")
NB = 29


def SPINUP(years):
    return DAY0 + pd.DateOffset(years=years)


def load_var(ddir, var):
    arrs, days = [], []
    for y in range(1979, 2027):
        f = Path(ddir) / f"{var}_{y}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        arrs.append(d["data"]); lat, lon = d["lat"], d["lon"]
        days.append(pd.Timestamp("1900-01-01") + pd.to_timedelta(d["time"], unit="h"))
    a = np.concatenate(arrs, 0)
    idx = pd.DatetimeIndex(np.concatenate([np.asarray(x) for x in days])).floor("D")
    full = pd.date_range(idx[0], idx[-1], freq="D")
    out = np.full((len(full),) + a.shape[1:], np.nan, np.float32)
    out[full.get_indexer(idx)] = a
    return full, out, lat, lon


def cell_bins(lat):
    b = H.lat_to_bin(lat)
    return np.where((lat > -46) & (lat < -17), b, -1)


def bin_daily_series(ddir):
    """Returns dict of DataFrames (index=day, columns=bin 0..28): P_land, P_andes, SWE_andes (degree-day model)."""
    cache = Path(ddir) / "bin_daily_v1.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        idx = pd.DatetimeIndex(z["days"])
        return {k: pd.DataFrame(z[k], index=idx, columns=range(NB)) for k in ("P_land", "P_andes", "SWE_andes")} | \
               {"meta": z["meta"].item()}
    days, P, lat, lon = load_var(ddir, "precip")
    el = np.load(Path(ddir) / "elevation.npz")["elev"]
    assert el.shape == P.shape[1:], (el.shape, P.shape)
    land = np.isfinite(P).mean(0) > 0.5
    cb = cell_bins(lat)
    Pf = np.where(np.isfinite(P), P, np.nan)
    Pl = np.full((len(days), NB), np.nan); Pa = Pl.copy(); meta = {}
    andes_masks = {}
    for b in range(NB):
        rows = cb == b
        m_land = land & rows[:, None]
        m_andes = m_land & (el >= 1500)
        if m_andes.sum() == 0:
            m_andes = m_land
        andes_masks[b] = m_andes
        Pl[:, b] = np.nanmean(Pf[:, m_land], axis=1)
        Pa[:, b] = np.nanmean(Pf[:, m_andes], axis=1)
        meta[b] = (int(m_land.sum()), int((m_land & (el >= 1500)).sum()))
    # snow: degree-day on Andes cells (daily Tmean=(tmax+tmin)/2)
    SWE = np.full((len(days), NB), np.nan)
    try:
        d2, TX, *_ = load_var(ddir, "tmax"); d3, TN, *_ = load_var(ddir, "tmin")
        common = days.intersection(d2).intersection(d3)
        ip, ix, inn = days.get_indexer(common), d2.get_indexer(common), d3.get_indexer(common)
        Tm = 0.5 * (TX[ix] + TN[inn])
        Pc = P[ip]
        for b in range(NB):
            m = andes_masks[b]
            pt = np.nan_to_num(Pc[:, m]); tt = Tm[:, m]
            # carry forward missing temperatures (causal) -> fallback 5C if none yet
            tt = pd.DataFrame(tt).ffill().fillna(5.0).to_numpy()
            swe = np.zeros(pt.shape[1]); ser = np.empty(len(common))
            for i in range(len(common)):
                swe = swe + pt[i] * (tt[i] <= 0)
                swe = np.maximum(0.0, swe - 3.0 * np.maximum(tt[i], 0))
                ser[i] = swe.mean()
            SWE[ip, b] = ser
    except Exception as e:
        print("SWE failed:", e)
    np.savez_compressed(cache, days=days.to_numpy(), P_land=Pl, P_andes=Pa, SWE_andes=SWE, meta=np.array(meta, dtype=object))
    return bin_daily_series(ddir)


def _exp_load(P, tau):
    """L[d] = L[d-1]*exp(-1/tau) + P[d]; NaN precip treated as 0 (rare)."""
    a = np.exp(-1.0 / tau)
    X = np.nan_to_num(np.asarray(P, float))
    L = np.empty_like(X); acc = np.zeros(X.shape[1])
    for i in range(len(X)):
        acc = acc * a + X[i]; L[i] = acc
    return L


def _sample_at(daily_vals, days, t_starts, lag_days=1):
    """value of a daily cumulative/state series at day t_start - lag_days - 1 (last fully allowed day)."""
    tgt = (t_starts - pd.Timedelta(days=lag_days + 1)).floor("D")
    ii = days.get_indexer(tgt)
    out = np.full((len(t_starts), daily_vals.shape[1]), np.nan)
    ok = ii >= 0
    out[ok] = daily_vals[ii[ok]]
    return out


def _window_sum(daily_vals, days, t_starts, window, lag_days=1):
    """sum over days D with t-lag-window <= D < t-lag (all NaN -> NaN)."""
    X = np.nan_to_num(np.asarray(daily_vals, float))
    C = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
    hi = days.searchsorted(t_starts - pd.Timedelta(days=lag_days), side="left")
    lo = days.searchsorted(t_starts - pd.Timedelta(days=lag_days + window), side="left")
    out = C[hi] - C[lo]
    full = (hi - lo) == window
    out[~full] = np.nan
    return out


def _anom(M, t_starts, horizon_days):
    """subtract bin x ISO-week climatology fitted on train issue times (t+h <= 2013-01-01)."""
    wk = np.minimum(t_starts.isocalendar().week.to_numpy().astype(int), 52)
    tr = (t_starts + pd.Timedelta(days=horizon_days)) <= TRAIN_END
    out = np.full_like(M, np.nan)
    for w in np.unique(wk):
        s = wk == w
        clim = np.nanmean(M[s & tr], axis=0)
        out[s] = M[s] - clim
    return out


def to_frame(t_starts, cols: dict):
    T, B = np.meshgrid(np.arange(len(t_starts)), np.arange(NB), indexing="ij")
    df = pd.DataFrame({"bin_id": B.ravel(), "t_start": t_starts[T.ravel()]})
    for k, v in cols.items():
        df[k] = v.ravel()
    return df


def panel_times(panel):
    return pd.DatetimeIndex(np.sort(panel.t_start.unique()))


def horizon_of(panel):
    return int(panel.attrs.get("horizon_days", 0)) or None


def feats_A(series, panel, h, P=None, days=None):
    days = series["P_land"].index if days is None else days
    P = series["P_land"].to_numpy() if P is None else P
    ts = panel_times(panel)
    L = _exp_load(P, 180.0)
    load = _sample_at(L, days, ts)
    load[ts < SPINUP(1)] = np.nan          # reservoir spin-up (starts at 0 on 1979-01-01)
    r30 = _window_sum(P, days, ts, 30)
    return to_frame(ts, {"x_load180_anom": _anom(load, ts, h), "x_rate30_anom": _anom(r30, ts, h)})


def feats_E(series, panel, h, P=None, days=None):
    days = series["P_andes"].index if days is None else days
    P = series["P_andes"].to_numpy() if P is None else P
    ts = panel_times(panel)
    load = _sample_at(_exp_load(P, 365.0), days, ts)
    load[ts < SPINUP(2)] = np.nan
    return to_frame(ts, {"x_andesload365_anom": _anom(load, ts, h)})


def feats_D(series, panel, h):
    days = series["SWE_andes"].index
    S = series["SWE_andes"].to_numpy()
    ts = panel_times(panel)
    now = _sample_at(S, days, ts)
    prev = _sample_at(S, days, ts - pd.Timedelta(days=30))
    now[ts < SPINUP(1)] = np.nan; prev[ts < SPINUP(1)] = np.nan
    return to_frame(ts, {"x_swe_anom": _anom(now, ts, h), "x_dswe30": now - prev})


def feats_B(panel, h):
    ts = panel_times(panel)
    mid = ts + pd.Timedelta(days=h / 2)
    ph = 2 * np.pi * (mid.dayofyear.to_numpy() - 1 + 0.5 * (h % 2 == 0)) / 365.25
    return pd.DataFrame({"t_start": ts, "x_season_sin": np.sin(ph), "x_season_cos": np.cos(ph)})


def _read_oni(path):
    seas = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    df = pd.read_csv(path, sep=r"\s+")
    df["m"] = df["SEAS"].map({s: i + 1 for i, s in enumerate(seas)})
    df["month"] = pd.to_datetime(dict(year=df["YR"], month=df["m"], day=1))
    # centered 3-month mean for month m uses m+1 -> available from day 10 of month m+2
    df["avail"] = df["month"] + pd.DateOffset(months=2) + pd.Timedelta(days=9)
    return df[["avail", "ANOM"]].rename(columns={"ANOM": "v"})


def _read_pdo(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 2 and p[0].isdigit() and len(p[0]) == 4:
            y = int(p[0])
            for m, v in enumerate(p[1:13], 1):
                v = float(v)
                if abs(v) < 90:
                    rows.append((pd.Timestamp(y, m, 1) + pd.DateOffset(months=2) + pd.Timedelta(days=9), v))
    return pd.DataFrame(rows, columns=["avail", "v"])


def _asof(df, ts):
    df = df.sort_values("avail")
    i = df["avail"].searchsorted(ts - pd.Timedelta(days=1), side="right") - 1   # known by t-1 day
    v = df["v"].to_numpy()
    return np.where(i >= 0, v[np.clip(i, 0, None)], np.nan)


def feats_C(ddir, panel):
    ts = panel_times(panel)
    oni, pdo = _read_oni(Path(ddir) / "oni.txt"), _read_pdo(Path(ddir) / "pdo.txt")
    o, p = _asof(oni, ts), _asof(pdo, ts)
    o12, p12 = _asof(oni, ts - pd.Timedelta(days=365)), _asof(pdo, ts - pd.Timedelta(days=365))
    return pd.DataFrame({"t_start": ts, "x_oni": o, "x_pdo": p, "x_doni12": o - o12, "x_dpdo12": p - p12})


def difference_frame(df, lag_steps):
    """causal lag_steps-step difference of every x_ column, per bin."""
    out = df.sort_values(["bin_id", "t_start"]).copy() if "bin_id" in df else df.sort_values("t_start").copy()
    cols = [c for c in out.columns if c.startswith("x_")]
    if "bin_id" in out:
        out[cols] = out.groupby("bin_id")[cols].diff(lag_steps)
    else:
        out[cols] = out[cols].diff(lag_steps)
    out = out.rename(columns={c: c + "_d" for c in cols})
    return out
