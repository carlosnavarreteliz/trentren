"""Build causal daily ocean-bottom-pressure (OBP) anomaly series per 1-deg bin, and panel features.

Usage: python -I build_features.py <data_dir> [fd|rq]
Writes <data_dir>/daily_L_<src>.parquet (days x 29 bins, hPa) and components, plus sea-level per station.
See PREREGISTRATION.md for definitions.
"""
import sys, os, json
import numpy as np
import pandas as pd

STATIONS = {"083": -18.4667, "080": -23.6533, "088": -27.0667, "081": -33.0267, "684": -41.485}
RHO_G_HPA_PER_MM = 1025 * 9.81 / 1000 / 100  # 1 mm water -> hPa  (=0.1006)


def load_hourly(fn):
    d = pd.read_csv(fn, header=None, names=["y", "m", "d", "h", "v"])
    d = d[(d.v > -32000)]
    t = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d))
    return pd.DataFrame({"day": t, "v": d.v.to_numpy(float)})


def daily_sea_level(fn, godin=False):
    h = load_hourly(fn)
    if godin:
        # causal (trailing) Godin filter A24*A24*A25 on a regular hourly grid; value at hour 23 of day D uses only
        # hours <= D 23:00. Requires all 71 hours valid.
        d = pd.read_csv(fn, header=None, names=["y", "m", "d", "h", "v"])
        t = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d, hour=d.h))
        x = pd.Series(np.where(d.v > -32000, d.v, np.nan), index=t)
        x = x[~x.index.duplicated()].asfreq("h")
        f = x.rolling(24, min_periods=24).mean().rolling(24, min_periods=24).mean().rolling(25, min_periods=25).mean()
        f = f[f.index.hour == 23]
        s = pd.Series(f.to_numpy(), index=f.index.floor("D"))
    else:
        g = h.groupby("day").v.agg(["mean", "count"])
        s = g["mean"].where(g["count"] >= 20)
    idx = pd.date_range("1970-01-01", "2026-10-17", freq="D")
    s = s.reindex(idx)
    # spike filter vs causal trailing 30-day median (days D-30..D-1)
    med = s.shift(1).rolling(30, min_periods=10).median()
    s = s.where(~((s - med).abs() > 500))
    return s


def causal_anom(s, win=365, minp=180):
    """x(D) - mean(x over D-win..D-1)."""
    return s - s.shift(1).rolling(win, min_periods=minp).mean()


def build(data_dir, src="fd", godin=False):
    idx = pd.date_range("1970-01-01", "2026-10-17", freq="D")
    eta = {}
    for sid, lat in STATIONS.items():
        fn = os.path.join(data_dir, f"h{sid}_{src}.csv")
        if not os.path.exists(fn):
            print("missing", fn); continue
        eta[sid] = daily_sea_level(fn, godin=godin)
    eta = pd.DataFrame(eta).reindex(idx)
    eta_a = eta.apply(causal_anom)
    # pressure per bin
    P = {}
    for b in range(29):
        j = json.load(open(os.path.join(data_dir, "era5", f"era5_sp_bin{b:02d}.json")))
        s = pd.Series(j["daily"]["surface_pressure_mean"], index=pd.to_datetime(j["daily"]["time"]), dtype=float)
        P[b] = s
    P = pd.DataFrame(P).reindex(idx)
    P_a = P.apply(causal_anom)
    # nearest available station by latitude for each bin, per day
    lat_c = np.arange(-45.5, -17.0, 1.0)
    sids = list(eta_a.columns)
    slat = np.array([STATIONS[s] for s in sids])
    E = eta_a.to_numpy()
    SL = np.full((len(idx), 29), np.nan)
    for b, la in enumerate(lat_c):
        order = np.argsort(np.abs(slat - la))
        for k in order[::-1]:  # fill farthest first, nearest overwrites
            if abs(slat[k] - la) > 6.0:
                continue
            v = E[:, k]
            SL[:, b] = np.where(np.isfinite(v), v, SL[:, b])
    SL = pd.DataFrame(SL, index=idx)
    L = RHO_G_HPA_PER_MM * SL + P_a.values
    out = {"L": L, "SL": RHO_G_HPA_PER_MM * SL, "PA": pd.DataFrame(P_a.values, index=idx)}
    return out, eta, eta_a


if __name__ == "__main__":
    data_dir = sys.argv[1]
    src = sys.argv[2] if len(sys.argv) > 2 else "fd"
    godin = len(sys.argv) > 3 and sys.argv[3] == "godin"
    out, eta, eta_a = build(data_dir, src, godin)
    if godin:
        src = src + "godin"
    for k, v in out.items():
        v.columns = [str(c) for c in v.columns]
        v.to_parquet(os.path.join(data_dir, f"daily_{k}_{src}.parquet"))
    eta.to_parquet(os.path.join(data_dir, f"eta_daily_{src}.parquet"))
    eta_a.to_parquet(os.path.join(data_dir, f"eta_anom_{src}.parquet"))
    print("coverage (fraction non-NaN 1976-2026) per station:")
    print(eta.loc["1976":].notna().mean().round(3))
    print("L coverage by bin:", out["L"].loc["1976":].notna().mean().round(2).to_dict())
