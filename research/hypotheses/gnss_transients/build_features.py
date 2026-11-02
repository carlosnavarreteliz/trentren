"""
Build causal per-station GNSS transient anomalies on the weekly harness issue grid.

usage: python -I build_features.py <data_dir> <out_dir>
  data_dir: contains tenv3/*.tenv3, steps.txt, chile_stations.csv (NGL downloads, untrusted -> parsed as data)
  out_dir : writes station_anoms.parquet

For each station s, issue time t (1976-01-01 + 7k days), lag L in {2, 14} and hinge window W in {30, 90}:
  data days D in [t-L-760, t-L)  (strictly before t-L)
  u = E sin78 + N cos78 (landward along convergence)
  OLS: u = a + b tau + annual + semiannual + sum_k o_k H(tau - tau_k) [NGL steps dated in window]
       + c * max(0, tau - (t-L-W))
  requirements: >=365 days before the hinge, >= W/2 days inside it.
  outliers > 5 robust sigma removed once.
  z = -c / se(c)   (positive = trenchward acceleration)
Also records whether an NGL earthquake step of M>=6.5 occurred in the 730 d before t-L (postseismic flag).
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
AZ = np.deg2rad(78.0)
MJD0 = pd.Timestamp("1858-11-17")
ISSUE = pd.date_range("1976-01-01", "2026-10-17", freq="7D")
ISSUE_MJD = ((ISSUE - MJD0).days).to_numpy()
LAGS = (2, 14)
WINS = (30, 90)
REFLEN = 760

MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def parse_ngl_date(s):
    yy, mon, dd = int(s[:2]), MON[s[2:5]], int(s[5:7])
    yyyy = 1900 + yy if yy > 80 else 2000 + yy
    return (pd.Timestamp(yyyy, mon, dd) - MJD0).days


def load_steps():
    rows = []
    with open(DATA / "steps.txt") as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            try:
                mjd = parse_ngl_date(p[1])
                typ = int(p[2])
            except Exception:
                continue
            mag = float(p[5]) if typ == 2 and len(p) >= 6 else np.nan
            rows.append((p[0], mjd, typ, mag))
    return pd.DataFrame(rows, columns=["sta", "mjd", "type", "mag"])


def load_tenv3(path):
    d = pd.read_csv(path, sep=r"\s+", header=0, usecols=[3, 7, 8, 9, 10], engine="c")
    d.columns = ["mjd", "e0", "e", "n0", "n"]
    d = d.dropna()
    E = (d.e0.to_numpy(float) + d.e.to_numpy(float)) * 1000.0  # mm
    N = (d.n0.to_numpy(float) + d.n.to_numpy(float)) * 1000.0
    mjd = d.mjd.to_numpy(int)
    u = E * np.sin(AZ) + N * np.cos(AZ)
    o = np.argsort(mjd)
    mjd, u = mjd[o], u[o]
    keep = np.r_[True, np.diff(mjd) > 0]
    return mjd[keep], u[keep]


def fit_one(tau, y, steps, hinge0):
    """tau in years relative to hinge start (negative before). returns c, se (mm/yr)."""
    w = 2 * np.pi * tau
    cols = [np.ones_like(tau), tau, np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)]
    # v2 fix: NGL lists several steps on the same/adjacent days (e.g. aftershocks). Identical Heaviside columns made
    # X'X singular -> NaN anomalies (~30 % of rows, concentrated after big sequences). Keep one column per distinct
    # split of the data.
    seen = set()
    for st in steps:
        h = (tau >= st).astype(float)
        k = int(h.sum())
        if 3 <= k <= len(h) - 3 and k not in seen:
            seen.add(k)
            cols.append(h)
    cols.append(np.maximum(0.0, tau - hinge0))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
    ok = np.abs(r) < 5 * mad
    if ok.sum() < len(y) and ok.sum() > X.shape[1] + 20 and X[ok, -1].max() > 0:
        X, y = X[ok], y[ok]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(y) - X.shape[1]
    if dof < 20:
        return np.nan, np.nan
    s2 = ((y - X @ beta) ** 2).sum() / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    v = cov[-1, -1]
    if not np.isfinite(v) or v <= 0 or np.linalg.cond(X) > 1e8:
        return np.nan, np.nan
    return beta[-1], np.sqrt(v)


def process_station(args):
    sta, lat, lon, st_df = args
    path = DATA / "tenv3" / f"{sta}.tenv3"
    if not path.exists():
        return None
    try:
        mjd, u = load_tenv3(path)
    except Exception:
        return None
    if len(mjd) < 400:
        return None
    step_mjd = st_df.mjd.to_numpy() if st_df is not None else np.array([], int)
    eq = st_df[(st_df.type == 2) & (st_df.mag >= 6.5)].mjd.to_numpy() if st_df is not None else np.array([], int)
    out = []
    first, last = mjd[0], mjd[-1]
    for ti in ISSUE_MJD:
        for L in LAGS:
            tend = ti - L                       # use days < tend
            if tend - 365 < first or tend - 60 > last:
                continue
            i0 = np.searchsorted(mjd, tend - REFLEN, "left")
            i1 = np.searchsorted(mjd, tend, "left")
            if i1 - i0 < 300:
                continue
            m = mjd[i0:i1]
            y = u[i0:i1]
            post = bool(((eq < tend) & (eq >= tend - 730)).any())
            rec = {"sta": sta, "lat": lat, "lon": lon, "issue_mjd": ti, "lag": L, "post65": post}
            for W in WINS:
                h0 = tend - W
                nb = (m < h0).sum(); na = len(m) - nb
                if nb < 365 or na < W // 2:
                    rec[f"c{W}"] = np.nan; rec[f"se{W}"] = np.nan
                    continue
                tau = (m - h0) / 365.25
                stp = [(s - h0) / 365.25 for s in step_mjd if (m[0] < s < tend)]
                c, se = fit_one(tau.astype(float), y.copy(), stp, 0.0)
                rec[f"c{W}"] = c; rec[f"se{W}"] = se
            rec["nobs"] = len(m)
            out.append(rec)
    return pd.DataFrame(out) if out else None


def main():
    stations = pd.read_csv(DATA / "chile_stations.csv")
    steps = load_steps()
    steps = steps[steps.sta.isin(stations.sta)]
    g = dict(tuple(steps.groupby("sta")))
    jobs = [(r.sta, r.lat, r.lon, g.get(r.sta)) for r in stations.itertuples()]
    with Pool(int(os.environ.get("NPROC", "2"))) as pool:
        res = [r for r in pool.imap_unordered(process_station, jobs, chunksize=2) if r is not None]
    df = pd.concat(res, ignore_index=True)
    df["t_start"] = MJD0 + pd.to_timedelta(df.issue_mjd, unit="D")
    for W in WINS:
        df[f"z{W}"] = np.clip(-df[f"c{W}"] / df[f"se{W}"], -10, 10)
    df.to_parquet(OUT / "station_anoms.parquet", index=False)
    print(df.shape, df.sta.nunique(), "stations")
    print(df.groupby("lag")[["z30", "z90"]].describe().T)


if __name__ == "__main__":
    main()
