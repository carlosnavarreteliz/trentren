"""Causal per-bin station-noise anomaly features for the harness panel.

build_site_z(long_parquet, channels_txt) -> DataFrame (date, site, lat, band, z, dense)
panel_features(site_z, panel, ...) -> DataFrame [bin_id, t_start, x_hf, x_lp, x_ms]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import harness as H  # noqa: E402

BANDS = {"hf": ([1.037, 3.2], (-200, -60)),
         "lp": ([0.005256, 0.009639], (-200, -80)),
         "ms": ([0.1, 0.3084], (-200, -60))}
LOC_PRIORITY = {"00": 0, "--": 1, "": 1, "10": 2, "60": 3}
BASE_LAG, BASE_LEN, MIN_BASE, SCALE_FLOOR, ZCLIP = 30, 335, 60, 1.0, 6.0


def read_channels(channels_txt):
    rows = []
    for line in Path(channels_txt).read_text().splitlines()[1:]:
        f = line.split("|")
        if len(f) < 17 or f[3] != "BHZ":
            continue
        rows.append(dict(net=f[0], sta=f[1], loc=f[2] if f[2] else "--", lat=float(f[4]),
                         start=pd.Timestamp(f[15][:10]), end=pd.Timestamp(f[16][:10]) if f[16] else pd.NaT))
    return pd.DataFrame(rows)


def stream_z(series: pd.Series, epoch_starts) -> pd.Series:
    """series: daily values (DatetimeIndex, may have gaps). Returns causal robust z per day."""
    s = series.sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full)
    ep = np.searchsorted(np.sort(np.asarray(epoch_starts, dtype="datetime64[ns]")), full.values, side="right")
    out = pd.Series(np.nan, index=full)
    for e in np.unique(ep):
        m = ep == e
        x = s[m]
        past = x.shift(BASE_LAG)  # value at D is from D-30; rolling covers D-364..D-30
        r = past.rolling(BASE_LEN, min_periods=MIN_BASE)
        med = r.median()
        iqr = r.quantile(0.75) - r.quantile(0.25)
        scale = np.maximum(iqr / 1.349, SCALE_FLOOR)
        out[m] = ((x - med) / scale).clip(-ZCLIP, ZCLIP)
    return out


def build_site_z(long_parquet, channels_txt) -> pd.DataFrame:
    df = pd.read_parquet(long_parquet)
    ch = read_channels(channels_txt)
    lat = ch.groupby(["net", "sta"]).lat.first()
    parts = []
    for tgt, d in df.groupby("target"):
        net, sta, loc, _cha, _q = tgt.split(".")
        loc = loc if loc else "--"
        eps = ch[(ch.net == net) & (ch.sta == sta) & (ch["loc"] == loc)].start.tolist()
        for band, (freqs, (lo, hi)) in BANDS.items():
            b = d[np.isin(d.freq.round(4), np.round(freqs, 4))]
            b = b[(b.db >= lo) & (b.db <= hi)]
            if b.empty:
                continue
            v = b.groupby("date").db.mean()
            # require every frequency of the band present that day
            nf = b.groupby("date").db.size()
            v = v[nf == len(freqs)]
            if len(v) < MIN_BASE + BASE_LAG:
                continue
            z = stream_z(v, eps).dropna()
            if z.empty:
                continue
            parts.append(pd.DataFrame({"date": z.index, "z": z.values, "band": band, "net": net, "sta": sta,
                                       "loc": loc, "prio": LOC_PRIORITY.get(loc, 9),
                                       "lat": float(lat.get((net, sta), np.nan))}))
    z = pd.concat(parts, ignore_index=True)
    z["site"] = z.sta  # one site per station code (PEL in G is one site across loc codes)
    z = z.sort_values(["site", "band", "date", "prio"]).drop_duplicates(["site", "band", "date"], keep="first")
    z["dense"] = z.net == "C1"
    return z.reset_index(drop=True)


def panel_features(site_z: pd.DataFrame, panel: pd.DataFrame, window_days=14, lag_days=1, min_days=5,
                   radius_deg=1.5, dense=False, bands=("hf", "lp", "ms")) -> pd.DataFrame:
    """Per (bin, t_start): mean over sites within radius of each site's mean z over [t-lag-window, t-lag)."""
    z = site_z if dense else site_z[~site_z.dense]
    t_issue = np.sort(panel.t_start.unique())
    t_issue = pd.DatetimeIndex(t_issue)
    centers = H.bin_centers()
    nb = len(centers)
    out = {}
    for band in bands:
        zb = z[z.band == band]
        acc = np.zeros((len(t_issue), nb)); cnt = np.zeros((len(t_issue), nb))
        for site, d in zb.groupby("site"):
            s = d.set_index("date").z.sort_index()
            full = pd.date_range(s.index.min(), s.index.max() + pd.Timedelta(days=window_days + lag_days + 1), freq="D")
            s = s.reindex(full)
            # r(D) = mean over D-window+1..D ; feature(t) = r(t - lag - 1)  -> days [t-lag-window, t-lag)
            r = s.rolling(window_days, min_periods=min_days).mean()
            f = r.reindex(t_issue - pd.Timedelta(days=lag_days + 1)).to_numpy()
            ok = np.isfinite(f)
            lat = d.lat.iloc[0]
            near = np.abs(centers - lat) <= radius_deg
            for b in np.where(near)[0]:
                acc[ok, b] += f[ok]; cnt[ok, b] += 1
        val = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        out[f"x_{band}"] = val
    T, B = np.meshgrid(np.arange(len(t_issue)), np.arange(nb), indexing="ij")
    res = pd.DataFrame({"t_start": t_issue[T.ravel()], "bin_id": B.ravel()})
    for k, v in out.items():
        res[k] = v.ravel()
    return res
