"""
Shared, leakage-safe evaluation harness for testing candidate earthquake predictors in Chile.

See HARNESS.md for the full documentation. Quick start:

    import sys; sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
    import harness as H
    panel = H.build_panel(step_days=7, horizon_days=7, target_mag=6.0)   # cached
    res = H.evaluate(my_features, target="y_all", model="logit", panel=panel)
    sur = H.surrogate_test(my_features, n=200, mode="circular_shift", panel=panel)

Conventions
-----------
* All times are UTC, stored as tz-naive pandas Timestamps (datetime64[ns]).
* Panel row = (bin_id, t_start). The forecast is issued at t_start and covers [t_start, t_start + horizon).
* Every baseline feature uses ONLY events with time < t_start.
* Train = rows with t_start + horizon <= 2013-01-01 (purged so the label window never reaches the test period).
  Test = rows with t_start >= 2013-01-01.
"""
from __future__ import annotations

import os
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtr

HARNESS_VERSION = "1.0"
ROOT = Path("/Users/cnavarreteliz/Documents/Repos/earthquakes")
CATALOG_PATH = ROOT / "data/raw/chile_usgs_1900_2026.csv"
CACHE_DIR = ROOT / "research/cache"

LAT_MIN, LAT_MAX = -46.0, -17.0
LON_MIN, LON_MAX = -78.0, -66.0
HISTORY_START = pd.Timestamp("1973-01-01")   # events used as history for features
PANEL_START = pd.Timestamp("1976-01-01")      # first forecast issue time
TEST_START = pd.Timestamp("2013-01-01")
M_MODEL = 4.7                                  # modeling threshold for feature events (see HARNESS.md: Mc)
N_HYPOTHESES = 14
ALPHA_BONF = 0.05 / N_HYPOTHESES

_EPOCH = pd.Timestamp("1970-01-01")


def to_days(ts) -> np.ndarray:
    """Timestamps (tz-naive UTC) -> float days since 1970-01-01."""
    ts = pd.to_datetime(ts)
    if isinstance(ts, pd.Timestamp):
        return np.array((ts - _EPOCH) / pd.Timedelta(days=1))
    return ((pd.DatetimeIndex(ts) - _EPOCH) / pd.Timedelta(days=1)).to_numpy(dtype=float)


# ----------------------------------------------------------------------------------------------
# 1. Catalog + completeness
# ----------------------------------------------------------------------------------------------
_CATALOG = None


def load_catalog(path: str | Path = CATALOG_PATH, start=HISTORY_START, min_mag: float | None = None) -> pd.DataFrame:
    """Load the USGS catalog, region lat -46..-17, lon -78..-66 (includes outer rise), type=='earthquake'.

    Returns columns: time (tz-naive UTC), t_days (days since 1970), latitude, longitude, depth, mag, magType, id,
    bin_id (1-degree latitude bin, 0 = -46..-45).  Sorted by time. Events from `start` (default 1973-01-01; the
    extra 3 years before the 1976 panel start are history for the features).
    """
    global _CATALOG
    if _CATALOG is None or str(path) != str(CATALOG_PATH):
        d = pd.read_csv(path)
        d = d[d["type"] == "earthquake"].copy()
        d["time"] = pd.to_datetime(d["time"], format="ISO8601", utc=True).dt.tz_convert(None)
        d = d[d.latitude.between(LAT_MIN, LAT_MAX) & d.longitude.between(LON_MIN, LON_MAX)]
        d = d.dropna(subset=["mag"]).sort_values("time").reset_index(drop=True)
        d["t_days"] = to_days(d["time"])
        d["bin_id"] = lat_to_bin(d["latitude"].to_numpy(), 1.0)
        keep = ["time", "t_days", "latitude", "longitude", "depth", "mag", "magType", "id", "bin_id"]
        if str(path) == str(CATALOG_PATH):
            _CATALOG = d[keep]
        else:
            out = d[keep]
            out = out[out.time >= pd.Timestamp(start)]
            return out if min_mag is None else out[out.mag >= min_mag].reset_index(drop=True)
    out = _CATALOG[_CATALOG.time >= pd.Timestamp(start)]
    if min_mag is not None:
        out = out[out.mag >= min_mag]
    return out.reset_index(drop=True)


def lat_to_bin(lat, bin_deg: float = 1.0) -> np.ndarray:
    """Latitude -> integer bin id (0 = southernmost bin starting at -46). Values outside [-46,-17] -> -1."""
    lat = np.asarray(lat, dtype=float)
    nb = int(round((LAT_MAX - LAT_MIN) / bin_deg))
    b = np.floor((lat - LAT_MIN) / bin_deg).astype(int)
    b = np.where(lat == LAT_MAX, nb - 1, b)
    return np.where((b >= 0) & (b < nb), b, -1)


def bin_centers(bin_deg: float = 1.0) -> np.ndarray:
    nb = int(round((LAT_MAX - LAT_MIN) / bin_deg))
    return LAT_MIN + bin_deg * (np.arange(nb) + 0.5)


def mc_maxc(mags, binw: float = 0.1, correction: float = 0.2) -> float:
    """Maximum-curvature magnitude of completeness (+ standard 0.2 correction, Woessner & Wiemer 2005)."""
    m = np.round(np.asarray(mags) / binw) * binw
    if len(m) < 30:
        return np.nan
    vals, cnt = np.unique(m, return_counts=True)
    return float(vals[np.argmax(cnt)] + correction)


def completeness_by_year(cat: pd.DataFrame | None = None, window_years: int = 3) -> pd.DataFrame:
    """Mc(maxc+0.2) per calendar year using a centered `window_years` window (descriptive only, not a feature)."""
    cat = load_catalog() if cat is None else cat
    years = np.arange(cat.time.dt.year.min(), cat.time.dt.year.max() + 1)
    rows = []
    for y in years:
        sel = cat[(cat.time.dt.year >= y - window_years // 2) & (cat.time.dt.year <= y + window_years // 2)]
        rows.append({"year": int(y), "n": int((cat.time.dt.year == y).sum()), "mc_maxc": mc_maxc(sel.mag),
                     "n_m45": int(((cat.time.dt.year == y) & (cat.mag >= 4.5)).sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# 2. Declustering
# ----------------------------------------------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def decluster(cat: pd.DataFrame | None = None, method: str = "zbz", min_mag: float = M_MODEL,
              b: float = 1.0, df: float = 1.6, eta0: float | None = None) -> pd.DataFrame:
    """Decluster the catalog (events with mag >= min_mag).

    method='zbz': Zaliapin & Ben-Zion (2013) nearest-neighbour proximity
        eta_ij = dt_years * r_km^df * 10^(-b m_i)  (parent i earlier than j, r floored at 1 km).
        log10(eta0) is set by a 2-component 1-D Gaussian mixture on log10(eta) (fit on events before 2013 only).
    method='gk': Gardner & Knopoff (1974) windows (largest-first; foreshock window = aftershock window/2... see doc).

    Returns a copy with columns:
        parent_idx, log_eta (zbz), cluster_id,
        bg        : True if the event is NOT strongly linked to any EARLIER event (CAUSAL: uses only past data).
        mainshock : True if the event is the largest of its cluster (NON-CAUSAL: a later, larger event demotes it to
                    'foreshock'). Use `mainshock` ONLY for targets, never as a feature input.
    """
    cat = load_catalog(min_mag=min_mag) if cat is None else cat[cat.mag >= min_mag].reset_index(drop=True)
    cat = cat.sort_values("time").reset_index(drop=True)
    n = len(cat)
    t = cat.t_days.to_numpy() / 365.25
    la, lo, m = cat.latitude.to_numpy(), cat.longitude.to_numpy(), cat.mag.to_numpy()
    if method == "zbz":
        parent = np.full(n, -1)
        log_eta = np.full(n, np.inf)
        chunk = 500
        for s in range(1, n, chunk):
            e = min(n, s + chunk)
            jj = np.arange(s, e)
            dt = t[jj][:, None] - t[None, :e]
            r = np.maximum(_haversine_km(la[jj][:, None], lo[jj][:, None], la[None, :e], lo[None, :e]), 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                le = np.log10(np.where(dt > 0, dt, np.nan)) + df * np.log10(r) - b * m[None, :e]
            le[np.isnan(le)] = np.inf
            # only i < j allowed
            mask = np.arange(e)[None, :] >= jj[:, None]
            le[mask] = np.inf
            k = np.argmin(le, axis=1)
            parent[jj] = np.where(np.isfinite(le[np.arange(len(jj)), k]), k, -1)
            log_eta[jj] = le[np.arange(len(jj)), k]
        if eta0 is None:
            from sklearn.mixture import GaussianMixture
            fit = log_eta[np.isfinite(log_eta) & (cat.time.to_numpy() < np.datetime64(TEST_START))]
            gm = GaussianMixture(2, random_state=0).fit(fit.reshape(-1, 1))
            grid = np.linspace(np.percentile(fit, 1), np.percentile(fit, 99), 2000).reshape(-1, 1)
            post = gm.predict_proba(grid)
            lo_comp = int(np.argmin(gm.means_.ravel()))
            mu = gm.means_.ravel()
            between = (grid.ravel() > mu.min()) & (grid.ravel() < mu.max())
            idx = np.where(between & (post[:, lo_comp] < 0.5))[0]
            eta0 = float(grid.ravel()[idx[0]]) if len(idx) else float(np.mean(mu))
        linked = np.isfinite(log_eta) & (log_eta < eta0)
        out = cat.copy()
        out["parent_idx"] = np.where(linked, parent, -1)
        out["log_eta"] = log_eta
        out.attrs["log10_eta0"] = eta0
    elif method == "gk":
        out = cat.copy()
        dkm = 10 ** (0.1238 * m + 0.983)
        tw = np.where(m >= 6.5, 10 ** (0.032 * m + 2.7389), 10 ** (0.5409 * m - 0.547))
        parent = np.full(n, -1)
        assigned = np.zeros(n, bool)
        td = cat.t_days.to_numpy()
        for i in np.argsort(-m, kind="stable"):
            if assigned[i]:
                continue
            dtt = td - td[i]
            win = (dtt <= tw[i]) & (dtt >= -tw[i] / 2.0) & (~assigned)
            win &= _haversine_km(la[i], lo[i], la, lo) <= dkm[i]
            win[i] = False
            parent[win] = i
            assigned[win] = True
            assigned[i] = True
        out["parent_idx"] = parent
        out["log_eta"] = np.nan
    else:
        raise ValueError(method)
    # clusters = connected components of the parent forest
    pid = out["parent_idx"].to_numpy()
    root = np.arange(n)
    for j in range(n):  # parents always point to earlier events for zbz; gk: parent is the window owner (a root)
        p = pid[j]
        root[j] = root[p] if p >= 0 else j
    if method == "gk":
        root = np.where(pid >= 0, pid, np.arange(n))
    out["cluster_id"] = root
    mag_rank = out.groupby("cluster_id")["mag"].transform("max")
    first_max = out[out.mag == mag_rank].groupby("cluster_id").head(1).index
    out["mainshock"] = False
    out.loc[first_max, "mainshock"] = True
    out["bg"] = pid < 0 if method == "zbz" else out["mainshock"]
    return out


_DECL = {}


def get_events(min_mag: float = M_MODEL, declustered: bool = False, start=HISTORY_START, end=None,
               method: str = "zbz") -> pd.DataFrame:
    """Event list (UTC). declustered=True keeps only mainshocks (largest event of each cluster).
    Columns: time, t_days, latitude, longitude, depth, mag, magType, id, bin_id, cluster_id, bg, mainshock.
    For min_mag < M_MODEL (4.7) declustering is not available (declustered must be False); M<4.7 is incomplete before ~1990.
    """
    if min_mag < M_MODEL:
        if declustered:
            raise ValueError(f"declustering is defined for min_mag >= {M_MODEL} only")
        ev = load_catalog(min_mag=min_mag)
    else:
        if method not in _DECL:
            _DECL[method] = decluster(method=method)
        ev = _DECL[method]
        ev = ev[ev.mag >= min_mag]
        if declustered:
            ev = ev[ev.mainshock]
    ev = ev[ev.time >= pd.Timestamp(start)]
    if end is not None:
        ev = ev[ev.time < pd.Timestamp(end)]
    return ev.reset_index(drop=True)


def schuster_test(phases_rad, weights=None) -> dict:
    """Schuster test for non-uniformity of phases (e.g. tidal phase at event times).
    p = exp(-R^2 / N). NOTE: valid only if the phase distribution expected under the null is uniform. For tides the
    phase-occupation is not exactly uniform; compare against phases of random times (see schuster_vs_background)."""
    ph = np.asarray(phases_rad, float)
    w = np.ones_like(ph) if weights is None else np.asarray(weights, float)
    n = len(ph)
    C, S = np.sum(w * np.cos(ph)), np.sum(w * np.sin(ph))
    R2 = C ** 2 + S ** 2
    neff = np.sum(w ** 2)
    p = float(np.exp(-R2 / neff)) if n else np.nan
    return {"n": n, "R": float(np.sqrt(R2)), "p_value": p,
            "mean_phase_rad": float(np.arctan2(S, C)), "amplitude": float(np.sqrt(R2) / np.sum(w)) if n else np.nan}


def schuster_vs_background(event_phases, background_phases, n_mc: int = 5000, seed: int = 0) -> dict:
    """Monte-Carlo Schuster test: compare event phases with phases sampled from a background (e.g. the phase at
    uniformly random times over the same period). Corrects for non-uniform phase occupation. Returns empirical p."""
    rng = np.random.default_rng(seed)
    ev = np.asarray(event_phases)
    bg = np.asarray(background_phases)
    # map to the background CDF so the null is uniform
    def stat(x):
        return np.abs(np.sum(np.exp(1j * x))) ** 2 / len(x)
    bgm = np.exp(1j * bg).mean()
    obs = np.abs(np.sum(np.exp(1j * ev) - bgm)) ** 2 / len(ev)
    null = np.empty(n_mc)
    for k in range(n_mc):
        s = rng.choice(bg, size=len(ev), replace=True)
        null[k] = np.abs(np.sum(np.exp(1j * s) - bgm)) ** 2 / len(ev)
    return {"n": len(ev), "stat": float(obs), "p_value": float((1 + np.sum(null >= obs)) / (n_mc + 1))}


# ----------------------------------------------------------------------------------------------
# 3. Panel + baseline features
# ----------------------------------------------------------------------------------------------
# temporal kernels (p, c[days]); productivity alpha; spatial floor s0 [deg]
ETAS_P_C = [(1.08, 0.02), (1.3, 0.02), (0.9, 0.5)]
ETAS_ALPHA = [0.8, 1.1]
ETAS_S0 = [0.3, 1.0]


def _rupture_halflen_deg(m):
    # Wells & Coppersmith (1994) subsurface rupture length, all slip types: log10 L(km) = -2.44 + 0.59 M
    return 0.5 * 10 ** (-2.44 + 0.59 * m) / 111.0


def _omori_integral(dt, h, p, c):
    """Integral over s in [t, t+h] of (s - t_e + c)^-p, dt = t - t_e > 0 (days)."""
    if abs(p - 1.0) < 1e-9:
        return np.log((dt + h + c) / (dt + c))
    return ((dt + c) ** (1 - p) - (dt + h + c) ** (1 - p)) / (p - 1)


def _spatial_weights(ev_lat, ev_mag, edges, s0):
    sig = np.sqrt(s0 ** 2 + _rupture_halflen_deg(ev_mag) ** 2)
    lo, hi = edges[:-1], edges[1:]
    return ndtr((hi[None, :] - ev_lat[:, None]) / sig[:, None]) - ndtr((lo[None, :] - ev_lat[:, None]) / sig[:, None])


def _etas_features(t_issue, ev, edges, h):
    """Omori-Utsu weighted sums of PAST events (te < t) integrated over the forecast horizon."""
    te = ev.t_days.to_numpy()
    me = ev.mag.to_numpy()
    lat = ev.latitude.to_numpy()
    nb = len(edges) - 1
    feats = {}
    rhs = {}
    for a in ETAS_ALPHA:
        w = 10 ** (a * (me - M_MODEL))
        for s0 in ETAS_S0:
            rhs[(a, s0)] = w[:, None] * _spatial_weights(lat, me, edges, s0)
    out = {}
    chunk = max(1, int(2e7 // max(1, len(te))))
    for (p, c) in ETAS_P_C:
        acc = {k: np.zeros((len(t_issue), nb)) for k in rhs}
        for s in range(0, len(t_issue), chunk):
            tt = t_issue[s:s + chunk]
            dt = tt[:, None] - te[None, :]
            K = np.where(dt > 0, _omori_integral(np.maximum(dt, 1e-9), h, p, c), 0.0)
            for k, R in rhs.items():
                acc[k][s:s + chunk] = K @ R
        for (a, s0), A in acc.items():
            out[f"etas_p{p}_c{c}_a{a}_s{s0}"] = A
    return out


def _count_features(t_issue, ev, nb, windows, nbr, prefix):
    out = {f"{prefix}_{w}d": np.zeros((len(t_issue), nb)) for w in windows}
    bins = ev.bin_id.to_numpy()
    te = ev.t_days.to_numpy()
    for b in range(nb):
        sel = np.abs(bins - b) <= nbr
        tb = np.sort(te[sel])
        hi = np.searchsorted(tb, t_issue, side="left")  # strictly before t_issue
        for w in windows:
            lo = np.searchsorted(tb, t_issue - w, side="left")
            out[f"{prefix}_{w}d"][:, b] = hi - lo
    return out


def _maxmag_features(t_issue, ev, nb, windows, nbr=1, floor=M_MODEL - 0.5):
    out = {f"maxmag_nbr_{w}d": np.full((len(t_issue), nb), floor) for w in windows}
    bins = ev.bin_id.to_numpy()
    for b in range(nb):
        sel = np.abs(bins - b) <= nbr
        tb = ev.t_days.to_numpy()[sel]
        mb = ev.mag.to_numpy()[sel]
        o = np.argsort(tb)
        tb, mb = tb[o], mb[o]
        hi = np.searchsorted(tb, t_issue, side="left")
        for w in windows:
            lo = np.searchsorted(tb, t_issue - w, side="left")
            arr = out[f"maxmag_nbr_{w}d"][:, b]
            for i in np.where(hi > lo)[0]:
                arr[i] = mb[lo[i]:hi[i]].max()
    return out


def _days_since(t_issue, ev, nb, min_mag, nbr=1, cap=365.25 * 50):
    out = np.full((len(t_issue), nb), cap)
    e = ev[ev.mag >= min_mag]
    for b in range(nb):
        tb = np.sort(e.t_days.to_numpy()[np.abs(e.bin_id.to_numpy() - b) <= nbr])
        idx = np.searchsorted(tb, t_issue, side="left") - 1
        ok = idx >= 0
        out[ok, b] = np.minimum(t_issue[ok] - tb[idx[ok]], cap)
    return out


def _cache_name(step_days, horizon_days, target_mag, bin_deg):
    return CACHE_DIR / f"panel_v{HARNESS_VERSION}_s{step_days}_h{horizon_days}_m{target_mag}_b{bin_deg}.parquet"


BASE_EXCLUDE = {"bin_id", "lat_center", "t_start", "y_all", "y_main", "n_all", "n_main", "split", "year",
                "p_poisson_all", "p_poisson_main"}


def build_panel(step_days: int = 7, horizon_days: int = 7, target_mag: float = 6.0, bin_deg: float = 1.0,
                use_cache: bool = True, verbose: bool = False) -> pd.DataFrame:
    """Space-time forecasting panel.

    Rows: latitude bins (bin_deg wide, -46..-17) x forecast issue times t_start = 1976-01-01 + k*step_days
    (last t_start such that t_start + horizon <= catalog end).
    Columns:
      bin_id, lat_center, t_start (tz-naive UTC), year, split ('train' | 'gap' | 'test')
      y_all  : 1 if any event M>=target_mag in the bin within [t_start, t_start+horizon)
      y_main : same, but only declustered mainshocks (ZBZ, largest event in cluster)
      n_all, n_main : counts (for reference)
      p_poisson_all / p_poisson_main : Poisson smoothed-seismicity background probability (training rates only)
      baseline features (all strictly from events with time < t_start, M>=4.5, from 1973):
        etas_* (12 Omori-Utsu/productivity/rupture-kernel sums, horizon-integrated), cnt_bin_{7,30,365}d,
        cnt_nbr_{30,365}d (bins within +-1), maxmag_nbr_{30,365}d, log_days_since_m6_nbr,
        log_bg_m45 (training-period M>=4.5 rate of bin, log), log_bg_target (training-period target rate, log)
    Raw etas/count columns are stored untransformed; the model applies log/log1p (see baseline_feature_cols()).
    """
    fn = _cache_name(step_days, horizon_days, target_mag, bin_deg)
    if use_cache and fn.exists():
        return pd.read_parquet(fn)
    t0 = time.time()
    ev_all = get_events(min_mag=M_MODEL)  # declustered flags included (ZBZ)
    edges = np.arange(LAT_MIN, LAT_MAX + 1e-9, bin_deg)
    nb = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    ev_all = ev_all.assign(bin_id=lat_to_bin(ev_all.latitude.to_numpy(), bin_deg))
    cat_end = load_catalog().time.max().floor("D")
    t_starts = pd.date_range(PANEL_START, cat_end - pd.Timedelta(days=horizon_days), freq=f"{step_days}D")
    ti = to_days(t_starts)
    h = float(horizon_days)

    feats = _etas_features(ti, ev_all, edges, h)
    feats.update(_count_features(ti, ev_all, nb, [7, 30, 365], 0, "cnt_bin"))
    feats.update(_count_features(ti, ev_all, nb, [30, 365], 1, "cnt_nbr"))
    feats.update(_maxmag_features(ti, ev_all, nb, [30, 365]))
    feats["log_days_since_m6_nbr"] = np.log10(1 + _days_since(ti, ev_all, nb, 6.0))

    # targets
    ys = {}
    for name, ev in [("all", ev_all[ev_all.mag >= target_mag]),
                     ("main", ev_all[(ev_all.mag >= target_mag) & ev_all.mainshock])]:
        cnt = np.zeros((len(ti), nb), int)
        for b in range(nb):
            tb = np.sort(ev.t_days.to_numpy()[ev.bin_id.to_numpy() == b])
            cnt[:, b] = np.searchsorted(tb, ti + h, side="left") - np.searchsorted(tb, ti, side="left")
        ys[name] = cnt

    T, B = np.meshgrid(np.arange(len(ti)), np.arange(nb), indexing="ij")
    panel = pd.DataFrame({"bin_id": B.ravel().astype(np.int16), "lat_center": centers[B.ravel()],
                          "t_start": t_starts[T.ravel()]})
    panel["year"] = panel.t_start.dt.year.astype(np.int16)
    end_t = panel.t_start + pd.Timedelta(days=horizon_days)
    panel["split"] = np.where(panel.t_start >= TEST_START, "test", np.where(end_t <= TEST_START, "train", "gap"))
    panel["y_all"] = (ys["all"].ravel() > 0).astype(np.int8)
    panel["y_main"] = (ys["main"].ravel() > 0).astype(np.int8)
    panel["n_all"] = ys["all"].ravel()
    panel["n_main"] = ys["main"].ravel()
    for k, v in feats.items():
        panel[k] = v.ravel().astype(np.float64)

    # training-period background rates (smoothed across bins), no test data used
    tr = panel.split == "train"
    ev_tr = ev_all[(ev_all.time >= PANEL_START) & (ev_all.time < TEST_START)]
    span_days = (TEST_START - PANEL_START).days
    rate45 = np.bincount(ev_tr.bin_id[ev_tr.bin_id >= 0], minlength=nb) / span_days * h
    panel["log_bg_m45"] = np.log10(_smooth_bins(rate45, 1.0) + 1e-4)[panel.bin_id]
    for tgt in ["all", "main"]:
        k = panel[tr].groupby("bin_id")[f"y_{tgt}"].agg(["sum", "count"]).reindex(range(nb), fill_value=0)
        # smoothed frequency with weak global prior (pseudo-count 1 event spread by global rate)
        glob = k["sum"].sum() / k["count"].sum()
        freq = (_smooth_bins(k["sum"].to_numpy().astype(float), 1.0) + glob) / (k["count"].to_numpy() + 1.0)
        panel[f"p_poisson_{tgt}"] = np.clip(freq, 1e-6, 0.5)[panel.bin_id]
    panel["log_bg_target"] = np.log10(panel["p_poisson_all"])
    panel.attrs = {}
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = fn.with_suffix(f".tmp{os.getpid()}")
        panel.to_parquet(tmp, index=False)
        os.replace(tmp, fn)
    if verbose:
        print(f"panel built in {time.time()-t0:.1f}s: {panel.shape}")
    return panel


def _smooth_bins(x, sigma_bins):
    n = len(x)
    i = np.arange(n)
    K = np.exp(-0.5 * ((i[:, None] - i[None, :]) / sigma_bins) ** 2)
    K /= K.sum(axis=0, keepdims=True)  # conserve totals
    return K @ x


def baseline_feature_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in BASE_EXCLUDE and not c.startswith("x_")]


def _transform_base(df: pd.DataFrame, cols) -> np.ndarray:
    X = np.empty((len(df), len(cols)))
    for j, c in enumerate(cols):
        v = df[c].to_numpy(dtype=float)
        if c.startswith("etas_"):
            v = np.log10(v + 1e-3)
        elif c.startswith("cnt_"):
            v = np.log1p(v)
        X[:, j] = v
    return X


# ----------------------------------------------------------------------------------------------
# 4/5. Models and evaluation
# ----------------------------------------------------------------------------------------------
_PANELS = {}


def get_panel(step_days=7, horizon_days=7, target_mag=6.0, bin_deg=1.0) -> pd.DataFrame:
    key = (step_days, horizon_days, float(target_mag), float(bin_deg))
    if key not in _PANELS:
        _PANELS[key] = build_panel(step_days, horizon_days, target_mag, bin_deg)
    return _PANELS[key]


def _ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def logit_newton(A: np.ndarray, y: np.ndarray, C: float = 1.0, w0: np.ndarray | None = None,
                 tol: float = 1e-10, max_iter: int = 60):
    """L2-penalized logistic regression, exact Newton with backtracking. Same objective as sklearn
    LogisticRegression(C=C): C*sum(logloss) + 0.5*||coef||^2, intercept unpenalized. Returns weights [coef..., intercept].
    Converges to the exact optimum (sklearn's lbfgs stops early on the collinear ETAS columns)."""
    n, d = A.shape
    Z = np.hstack([A, np.ones((n, 1))])
    y = y.astype(float)
    w = np.zeros(d + 1) if w0 is None else np.asarray(w0, float).copy()
    reg = np.r_[np.ones(d), 0.0] / C

    def obj(w):
        z = Z @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(reg * w * w)

    f = obj(w)
    for _ in range(max_iter):
        pz = 1.0 / (1.0 + np.exp(-(Z @ w)))
        g = Z.T @ (pz - y) + reg * w
        Hm = (Z * (pz * (1 - pz))[:, None]).T @ Z + np.diag(reg + 1e-12)
        step = np.linalg.solve(Hm, g)
        t = 1.0
        while True:
            wn = w - t * step
            fn = obj(wn)
            if fn <= f + 1e-12 or t < 1e-6:
                break
            t *= 0.5
        dec = f - fn
        w, f = wn, fn
        if dec < tol * max(1.0, abs(f)):
            break
    return w


def _fit_predict(Xtr, ytr, Xte, model, seed=0, w0=None):
    """Returns (p_test, p_train, coef). For logit, coef = standardized weights [coef..., intercept]."""
    if model == "logit":
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1
        A, Bm = (Xtr - mu) / sd, (Xte - mu) / sd
        A = np.nan_to_num(A)
        Bm = np.nan_to_num(np.clip(Bm, -10, 10))  # guard against extreme extrapolation
        w = logit_newton(A, ytr, C=1.0, w0=w0)
        sig = lambda M: 1.0 / (1.0 + np.exp(-(M @ w[:-1] + w[-1])))
        return sig(Bm), sig(A), w
    elif model == "gbm":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.02, num_leaves=7, min_child_samples=300,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=seed, n_jobs=2, verbose=-1, deterministic=True, force_row_wise=True)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1], clf.predict_proba(Xtr)[:, 1], clf.feature_importances_
    raise ValueError(model)


def molchan_ass(y, p, measure=None) -> float:
    """Molchan area skill score = 1 - area under the (tau, nu) curve; 0.5 = random, 1 = perfect.
    tau = fraction of the reference measure (default: uniform over cells) under alarm; nu = miss rate."""
    y = np.asarray(y)
    if y.sum() == 0:
        return np.nan
    w = np.ones(len(y)) if measure is None else np.asarray(measure, float)
    o = np.argsort(-np.asarray(p), kind="stable")
    tau = np.concatenate([[0], np.cumsum(w[o]) / w.sum()])
    nu = np.concatenate([[1], 1 - np.cumsum(y[o]) / y.sum()])
    return float(1 - np.trapezoid(nu, tau))


def rhoades_ttest(y, p_a, p_b) -> dict:
    """Paired T-test of Rhoades et al. (2011) adapted to Bernoulli cells (p used as the expected count).
    I_i = ln(pA_i/pB_i) - (sum pA - sum pB)/N over the N target cells. Positive = A better than B.
    Assumes independent target events -> anti-conservative under clustering; prefer surrogate p-values."""
    y = np.asarray(y).astype(bool)
    N = int(y.sum())
    if N < 2:
        return {"N": N, "IG": np.nan, "T": np.nan, "p_one_sided": np.nan}
    I = np.log(np.clip(p_a[y], 1e-12, 1) / np.clip(p_b[y], 1e-12, 1)) - (p_a.sum() - p_b.sum()) / N
    m, s = I.mean(), I.std(ddof=1)
    T = m / (s / np.sqrt(N)) if s > 0 else np.nan
    return {"N": N, "IG": float(m), "T": float(T), "p_one_sided": float(stats.t.sf(T, N - 1)) if s > 0 else np.nan}


def _metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score
    return {"ll": float(_ll(y, p).sum()), "brier": float(np.mean((p - y) ** 2)),
            "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
            "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else np.nan,
            "molchan_ass": molchan_ass(y, p), "expected": float(p.sum()), "observed": int(y.sum())}


def _prepare_extra(panel, extra, extra_cols=None):
    """Align extra features to the panel. Accepts frames keyed by [bin_id, t_start] or by [t_start] only
    (global series, broadcast to all bins). Returns (matrix aligned to panel rows, column names)."""
    if extra is None:
        return np.empty((len(panel), 0)), []
    e = extra.copy()
    if "t_start" not in e.columns:
        raise ValueError("extra_features needs a 't_start' column (and optionally 'bin_id')")
    e["t_start"] = pd.to_datetime(e["t_start"])
    if getattr(e["t_start"].dt, "tz", None) is not None:
        e["t_start"] = e["t_start"].dt.tz_convert(None)
    keys = ["bin_id", "t_start"] if "bin_id" in e.columns else ["t_start"]
    cols = extra_cols or [c for c in e.columns if c not in ("bin_id", "t_start", "lat_center")]
    if e.duplicated(keys).any():
        raise ValueError(f"extra_features has duplicate keys {keys}")
    e["t_start"] = e["t_start"].astype("datetime64[ns]")
    if "bin_id" in keys:
        e["bin_id"] = e["bin_id"].astype(panel["bin_id"].dtype)
    left = panel[["bin_id", "t_start"]].copy()
    left["t_start"] = left["t_start"].astype("datetime64[ns]")
    m = left.merge(e[keys + cols], on=keys, how="left")
    X = m[cols].to_numpy(dtype=float)
    frac_nan = np.mean(~np.isfinite(X), axis=0)
    if np.any(frac_nan > 0.999):
        raise ValueError(f"extra feature(s) {[c for c, f in zip(cols, frac_nan) if f > 0.999]} did not align with the "
                         "panel (all NaN after merge on t_start/bin_id). Check t_start values/time zone.")
    if np.any(frac_nan > 0.2):
        warnings.warn(f"extra features with >20% missing after alignment: "
                      f"{dict((c, round(float(f), 3)) for c, f in zip(cols, frac_nan) if f > 0.2)}")
    return X, cols


_BASE_CACHE = {}


def fit_etas_additive(E: np.ndarray, r_bg: np.ndarray, y: np.ndarray):
    """Additive ETAS-style rate for a Bernoulli cell: rate = w0*r_bg + sum_k w_k*E_k (all w >= 0),
    P(y=1) = 1 - exp(-rate). E = horizon-integrated Omori/productivity/rupture-kernel sums (etas_* columns),
    r_bg = -log(1 - p_poisson). Fitted by maximum likelihood (L-BFGS-B, bounds w>=0). Returns weights."""
    from scipy.optimize import minimize
    Z = np.column_stack([r_bg, E])
    sc = Z.mean(0)
    sc[sc == 0] = 1
    Zs = Z / sc
    yb = y.astype(float)

    def f(w):
        r = np.maximum(Zs @ w, 1e-12)
        q = -np.expm1(-r)
        ll = np.sum(yb * np.log(np.maximum(q, 1e-15)) - (1 - yb) * r)
        g_r = yb * np.exp(-r) / np.maximum(q, 1e-15) - (1 - yb)
        return -ll, -(Zs.T @ g_r)

    w0 = np.zeros(Z.shape[1])
    w0[0] = 1.0 / sc[0] * sc[0]
    w0[1:] = 1e-3
    w0[0] = np.mean(r_bg) / sc[0] * sc[0] / max(np.mean(Zs[:, 0]), 1e-12) * 0 + 1.0
    sol = minimize(f, w0, jac=True, method="L-BFGS-B", bounds=[(0, None)] * Z.shape[1],
                   options={"maxiter": 2000})
    return sol.x / sc


def etas_rate(panel: pd.DataFrame, w: np.ndarray, target: str) -> np.ndarray:
    tgt = "all" if target == "y_all" else "main"
    ecols = [c for c in panel.columns if c.startswith("etas_")]
    r_bg = -np.log1p(-panel[f"p_poisson_{tgt}"].to_numpy())
    return np.column_stack([r_bg, panel[ecols].to_numpy()]) @ w


def _baseline(panel, target, model):
    key = (id(panel), len(panel), target, model)
    if key in _BASE_CACHE:
        return _BASE_CACHE[key]
    cols = baseline_feature_cols(panel)
    Xb = _transform_base(panel, cols)
    tr = (panel.split == "train").to_numpy()
    te = (panel.split == "test").to_numpy()
    y = panel[target].to_numpy()
    # (1) pure additive ETAS-style model (MLE on train)
    ecols = [c for c in panel.columns if c.startswith("etas_")]
    tgt = "all" if target == "y_all" else "main"
    r_bg = -np.log1p(-panel[f"p_poisson_{tgt}"].to_numpy())
    w = fit_etas_additive(panel[ecols].to_numpy()[tr], r_bg[tr], y[tr])
    rate = etas_rate(panel, w, target)
    p_etas = -np.expm1(-np.maximum(rate, 1e-12))
    # (2) baseline = logit/gbm on all clustering features + log(additive ETAS rate)
    Xb = np.column_stack([Xb, np.log10(np.maximum(rate, 1e-8))])
    cols = cols + ["log_etas_rate"]
    p_te, p_tr, coef = _fit_predict(Xb[tr], y[tr], Xb[te], model)
    out = dict(cols=cols, Xb=Xb, tr=tr, te=te, y=y, p_base_te=p_te, p_base_tr=p_tr, p_etas_te=p_etas[te], coef=coef,
               etas_weights=w)
    _BASE_CACHE[key] = out
    return out


def evaluate(extra_features: pd.DataFrame | None = None, target: str = "y_all", model: str = "logit",
             panel: pd.DataFrame | None = None, extra_cols: list[str] | None = None,
             return_preds: bool = False, **panel_kw) -> dict:
    """Fit baseline-only and baseline+extra on train (t_start+horizon <= 2013-01-01), predict test (t_start >= 2013).

    extra_features: DataFrame with 't_start' (+ optional 'bin_id') and one or more numeric feature columns, computed
        ONLY from information available before t_start. Missing values are imputed as the train mean (0 after
        standardization) for logit; LightGBM handles NaN natively.
    target: 'y_all' or 'y_main'.  model: 'logit' (L2, C=1, standardized) or 'gbm' (LightGBM, fixed params).
    panel: a frame from build_panel(); if None, get_panel(**panel_kw) (defaults step=7, horizon=7, M>=6.0).

    Returns dict with:
      n_test_events (= positive test cells), n_train_events, n_test_cells,
      ig_per_event      : (LL_full - LL_base) / n_test_events  [nats per target event]  <-- PRIMARY
      ig_base_vs_poisson, ig_full_vs_poisson : same vs Poisson smoothed-seismicity background
      base, full, poisson : dicts with ll, brier, roc_auc, pr_auc, molchan_ass, expected, observed
      ttest_full_vs_base : Rhoades et al. (2011)-style paired T-test
      coef_extra (logit standardized coefs) or importance_extra (gbm split counts)
      preds (optional DataFrame: bin_id, t_start, year, y, p_base, p_full, p_poisson)
    """
    panel = get_panel(**panel_kw) if panel is None else panel
    base = _baseline(panel, target, model)
    tr, te, y = base["tr"], base["te"], base["y"]
    Xe, ecols = _prepare_extra(panel, extra_features, extra_cols)
    p_base = base["p_base_te"]
    if len(ecols):
        X = np.hstack([base["Xb"], Xe])
        if model == "logit":
            # impute NaN with train mean
            mu = np.nanmean(X[tr], axis=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            bad = ~np.isfinite(X)
            if bad.any():
                X = np.where(bad, mu[None, :], X)
        w0 = None
        if model == "logit":  # warm start: baseline weights + 0 for the extra columns (same standardization)
            w0 = np.r_[base["coef"][:-1], np.zeros(len(ecols)), base["coef"][-1]]
        p_full, _, coef = _fit_predict(X[tr], y[tr], X[te], model, w0=w0)
        if model == "logit":
            coef = coef[:-1]  # drop intercept so coef[-len(ecols):] are the extra features
    else:
        p_full, coef = p_base, np.array([])
    yt = y[te]
    tgt = "all" if target == "y_all" else "main"
    p_pois = panel[f"p_poisson_{tgt}"].to_numpy()[te]
    N = int(yt.sum())
    p_etas = base["p_etas_te"]
    mb, mf, mp, me = _metrics(yt, p_base), _metrics(yt, p_full), _metrics(yt, p_pois), _metrics(yt, p_etas)
    res = {
        "target": target, "model": model, "n_test_events": N, "n_train_events": int(y[tr].sum()),
        "n_test_cells": int(te.sum()), "extra_cols": ecols,
        "ig_per_event": (mf["ll"] - mb["ll"]) / N if N else np.nan,
        "ig_base_vs_poisson": (mb["ll"] - mp["ll"]) / N if N else np.nan,
        "ig_full_vs_poisson": (mf["ll"] - mp["ll"]) / N if N else np.nan,
        "ig_etas_vs_poisson": (me["ll"] - mp["ll"]) / N if N else np.nan,
        "ig_full_vs_etas": (mf["ll"] - me["ll"]) / N if N else np.nan,
        "base": mb, "full": mf, "poisson": mp, "etas": me,
        "ttest_full_vs_base": rhoades_ttest(yt, p_full, p_base),
        "ttest_base_vs_poisson": rhoades_ttest(yt, p_base, p_pois),
    }
    if len(ecols):
        k = "coef_extra" if model == "logit" else "importance_extra"
        res[k] = dict(zip(ecols, np.asarray(coef[-len(ecols):], float).round(4).tolist()))
    if return_preds:
        res["preds"] = pd.DataFrame({"bin_id": panel.bin_id.to_numpy()[te], "t_start": panel.t_start.to_numpy()[te],
                                     "year": panel.year.to_numpy()[te], "y": yt, "p_base": p_base,
                                     "p_full": p_full, "p_poisson": p_pois,
                                     "p_etas": p_etas})
    return res


def block_bootstrap_ci(preds: pd.DataFrame, a: str = "p_full", b: str = "p_base", n_boot: int = 2000,
                       seed: int = 0, ci: float = 0.95) -> dict:
    """Block bootstrap by calendar year of the test-period information gain per event of model a over model b.
    preds = evaluate(..., return_preds=True)['preds']."""
    rng = np.random.default_rng(seed)
    d = (_ll(preds.y.to_numpy(), preds[a].to_numpy()) - _ll(preds.y.to_numpy(), preds[b].to_numpy()))
    g = pd.DataFrame({"year": preds.year.to_numpy(), "d": d, "y": preds.y.to_numpy()}).groupby("year").sum()
    D, Y = g.d.to_numpy(), g.y.to_numpy()
    k = len(g)
    idx = rng.integers(0, k, size=(n_boot, k))
    num, den = D[idx].sum(1), Y[idx].sum(1)
    ig = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    lo, hi = np.nanpercentile(ig, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return {"ig_per_event": float(D.sum() / max(Y.sum(), 1)), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_boot_le0": float(np.mean(ig <= 0)), "n_years": int(k)}


# ----------------------------------------------------------------------------------------------
# 6. Surrogates
# ----------------------------------------------------------------------------------------------
def _full_grid(panel, extra, cols):
    """Extra features on the full (time x bin) grid of the panel, as an array [T, B, F]."""
    X, cols = _prepare_extra(panel, extra, cols)
    T = panel.t_start.nunique()
    B = panel.bin_id.nunique()
    if len(panel) != T * B or not (panel.bin_id.to_numpy()[:B] == np.arange(B)).all():
        raise ValueError("surrogate_test needs the full panel from get_panel()/build_panel() in its original row order")
    return X.reshape(T, B, -1), cols  # panel is ordered by (t, bin)


def _grid_to_frame(panel, G, cols):
    T, B, F = G.shape
    df = panel[["bin_id", "t_start"]].copy()
    for j, c in enumerate(cols):
        df[c] = G[:, :, j].ravel()
    return df


def _phase_rand(Z: np.ndarray, ph_full: np.ndarray) -> np.ndarray:
    F = np.fft.rfft(Z, axis=0)
    ph = ph_full[:F.shape[0]].copy()
    ph[0] = 0.0                          # keep the mean (levels) / net drift (increments)
    if Z.shape[0] % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(F * np.exp(1j * ph)[:, None], n=Z.shape[0], axis=0)


def ft_surrogate(x: np.ndarray, rng: np.random.Generator, axis: int = 0, method: str = "auto",
                 r1_threshold: float = 0.99) -> np.ndarray:
    """Fourier phase-randomized surrogate (Theiler et al. 1992), multivariate (Prichard & Theiler 1994).

    x: 1-D array or 2-D array [time, columns] (or 3-D [time, bins, features]); time on `axis`.
    The SAME random phases are applied to every column, so cross-correlations between columns/bins are preserved.
    method per column:
      'levels'     : classic FT surrogate of the levels (keeps mean + power spectrum). Right for stationary series.
      'increments' : FT surrogate of the first differences, re-integrated from the original start value (keeps the
                     increment spectrum, start value and net drift; no wrap-around jump). Right for random-walk-like
                     series, for which a levels surrogate would be a periodic "bridge" with no drift.
      'auto'       : 'increments' if the lag-1 autocorrelation of the column is > r1_threshold (0.99), else 'levels'.
    NaNs are filled by linear interpolation for generation and put back at the same positions afterwards.
    """
    x = np.asarray(x, float)
    one_d = x.ndim == 1
    Xm = x.reshape(len(x), -1) if one_d else np.moveaxis(x, axis, 0)
    shape = Xm.shape
    X = Xm.reshape(shape[0], -1).copy()
    T = len(X)
    nan = ~np.isfinite(X)
    if nan.any():
        idx = np.arange(T)
        for j in np.where(nan.any(0))[0]:
            ok = ~nan[:, j]
            X[:, j] = np.interp(idx, idx[ok], X[ok, j]) if ok.sum() >= 2 else 0.0
    if method == "auto":
        Xc = X - X.mean(0)
        den = np.sum(Xc ** 2, 0)
        r1 = np.where(den > 0, np.sum(Xc[1:] * Xc[:-1], 0) / np.where(den > 0, den, 1), 0.0)
        inc = r1 > r1_threshold
    else:
        inc = np.full(X.shape[1], method == "increments")
    ph_full = rng.uniform(0, 2 * np.pi, size=T // 2 + 1)
    Y = np.empty_like(X)
    if (~inc).any():
        Y[:, ~inc] = _phase_rand(X[:, ~inc], ph_full)
    if inc.any():
        ds = _phase_rand(np.diff(X[:, inc], axis=0), ph_full)
        Y[:, inc] = np.vstack([np.zeros((1, inc.sum())), np.cumsum(ds, axis=0)]) + X[:1, inc]
    Y[nan] = np.nan
    Y = Y.reshape(shape)
    if one_d:
        return Y.ravel()
    return np.moveaxis(Y, 0, axis)


def _end_matched_roll(G: np.ndarray, k: int) -> np.ndarray:
    """Circular shift along axis 0 after removing the straight line joining the first and last finite values of each
    series (then adding it back). This removes the artificial jump at the wrap point (important for random-walk-like or
    trending series) and keeps any linear trend aligned in time, so a feature gets no credit for a trend alone."""
    T = G.shape[0]
    t = np.arange(T, dtype=float)
    flat = G.reshape(T, -1)
    out = np.empty_like(flat)
    for j in range(flat.shape[1]):
        x = flat[:, j]
        ok = np.isfinite(x)
        if ok.sum() < 2:
            out[:, j] = np.roll(x, k)
            continue
        i0, i1 = np.argmax(ok), T - 1 - np.argmax(ok[::-1])
        ramp = x[i0] + (x[i1] - x[i0]) * (t - i0) / max(i1 - i0, 1)
        out[:, j] = np.roll(x - ramp, k) + ramp
    return out.reshape(G.shape)


def circular_shift_series(series: pd.Series, shift_days: int, end_matched: bool = True) -> pd.Series:
    """Circularly shift a raw (daily) series by shift_days positions, end-matched by default (see _end_matched_roll).
    Use inside a surrogate `builder` before re-aggregating with align_daily_series."""
    v = series.to_numpy(dtype=float)
    r = _end_matched_roll(v[:, None], int(shift_days))[:, 0] if end_matched else np.roll(v, int(shift_days))
    return pd.Series(r, index=series.index, name=series.name)


def surrogate_test(extra_features: pd.DataFrame | None = None, builder=None, n: int = 200,
                   mode: str = "phase_randomize", target: str = "y_all", model: str = "logit",
                   panel: pd.DataFrame | None = None, extra_cols=None, min_shift_days: int = 365,
                   seed: int = 0, n_jobs: int = 1, end_matched: bool = True, **panel_kw) -> dict:
    """Null distribution of ig_per_event under surrogates that destroy the feature-target alignment but keep the
    feature's own autocorrelation / spatial structure.

    mode='circular_shift': every feature column is circularly shifted along the panel time axis by the same random
        offset s (|s| >= min_shift_days and <= T-min_shift_days), identical for all bins (keeps spatial structure).
        end_matched=True (default): the straight line joining each series' first and last values is removed before
        rolling and added back after, so there is no artificial jump at the wrap point and linear trends stay
        aligned in time (a pure trend earns no significance). Plain np.roll was mis-calibrated for random walks
        on y_main (4/20 false positives at p<0.05) -- see HARNESS.md section 9b.
    mode='phase_randomize' (DEFAULT, PRIMARY): multivariate FT-increment surrogates (see ft_surrogate): keeps the
        autocorrelation/spectrum, drift and cross-bin structure, no wrap-around discontinuity. Calibrated for both
        stationary and random-walk-like (slow, non-stationary) features.
        (circular_shift is a SECONDARY check; with end_matched=False the wrap-around jump becomes a step regressor
        that overfits the train period, biasing the null IG downward -> anti-conservative for slow series.)
    mode='bin_permute' : bin labels of the feature are randomly permuted (same permutation at all times).
        Only meaningful for features that differ between bins.
    builder: alternatively a callable that rebuilds the feature frame from a surrogate of your RAW series:
        mode='phase_randomize': builder(rng) with rng a numpy Generator, or builder(None) for the REAL features.
            Inside, use H.ft_surrogate(raw_values, rng) on the raw daily series, then aggregate as usual.
        mode='circular_shift': builder(shift_days:int); builder(0) gives the real features.
    Returns dict: ig_obs, p_value (one-sided, (1+#null>=obs)/(n+1)), null_mean, null_sd, null_q95, null (list),
                  significant_0.05, significant_bonferroni (p < 0.05/14).
    """
    panel = get_panel(**panel_kw) if panel is None else panel
    rng = np.random.default_rng(seed)
    T = panel.t_start.nunique()
    step = (panel.t_start.iloc[panel.bin_id.nunique()] - panel.t_start.iloc[0]).days
    min_k = int(math.ceil(min_shift_days / step))
    if T - 2 * min_k <= 0:
        raise ValueError("panel too short for the requested min shift")
    if builder is not None:
        extra_features = builder(None if mode == "phase_randomize" else 0)
    obs = evaluate(extra_features, target, model, panel, extra_cols)["ig_per_event"]
    G, cols = _full_grid(panel, extra_features, extra_cols)
    shifts = rng.integers(min_k, T - min_k + 1, size=n)
    perms = [rng.permutation(G.shape[1]) for _ in range(n)]

    seeds = rng.integers(0, 2 ** 31 - 1, size=n)

    def one(k):
        if builder is not None and mode == "phase_randomize":
            ef = builder(np.random.default_rng(seeds[k]))
        elif builder is not None:
            ef = builder(int(shifts[k] * step))
        elif mode == "phase_randomize":
            ef = _grid_to_frame(panel, ft_surrogate(G, np.random.default_rng(seeds[k])), cols)
        elif mode == "circular_shift":
            ef = _grid_to_frame(panel, _end_matched_roll(G, int(shifts[k])) if end_matched
                                else np.roll(G, shifts[k], axis=0), cols)
        elif mode == "bin_permute":
            ef = _grid_to_frame(panel, G[:, perms[k], :], cols)
        else:
            raise ValueError(mode)
        return evaluate(ef, target, model, panel, cols)["ig_per_event"]

    if n_jobs > 1:
        from joblib import Parallel, delayed
        null = np.array(Parallel(n_jobs=min(n_jobs, 2), backend="threading")(delayed(one)(k) for k in range(n)))
    else:
        null = np.array([one(k) for k in range(n)])
    p = (1 + np.sum(null >= obs)) / (n + 1)
    return {"ig_obs": float(obs), "p_value": float(p), "null_mean": float(np.mean(null)),
            "null_sd": float(np.std(null)), "null_q95": float(np.quantile(null, 0.95)), "n": n, "mode": mode,
            "significant_0.05": bool(p < 0.05), "significant_bonferroni": bool(p < ALPHA_BONF),
            "null": null.round(5).tolist()}


# ----------------------------------------------------------------------------------------------
# Helpers for hypothesis agents
# ----------------------------------------------------------------------------------------------
def align_daily_series(series: pd.Series, panel: pd.DataFrame | None = None, window_days: int = 30,
                       agg: str = "mean", lag_days: int = 1, name: str | None = None) -> pd.DataFrame:
    """Aggregate a DAILY exogenous series to panel issue times WITHOUT look-ahead.

    series: pd.Series indexed by UTC calendar date (value for day D summarizes D 00:00..24:00 UTC). Sub-daily input
        is first averaged per UTC day.
    Window used for issue time t_start:  days D with  t_start - lag_days - window_days <= D < t_start - lag_days.
        lag_days=0 -> the last day used is the day that ended exactly at t_start (fully elapsed, no look-ahead).
        lag_days=1 (default) -> one extra day of safety margin (last day used = t_start - 2 days). Increase it for
        series published with delay (e.g. final/definitive products).
    Example: t_start=1976-01-01, window_days=7, lag_days=1 -> days 1975-12-24 .. 1975-12-30.
    agg: 'mean' | 'sum' | 'max' | 'min' | 'last' | 'std'.
    Returns DataFrame [t_start, <name>] (global series -> broadcast to all bins by evaluate()).
    """
    panel = get_panel() if panel is None else panel
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert(None)
    s = s.groupby(s.index.floor("D")).mean().sort_index()
    ts = pd.DatetimeIndex(np.sort(panel.t_start.unique()))
    d = s.index.to_numpy()
    v = s.to_numpy(dtype=float)
    out = np.full(len(ts), np.nan)
    hi_t = (ts - pd.Timedelta(days=lag_days)).to_numpy()
    lo_t = (ts - pd.Timedelta(days=lag_days + window_days)).to_numpy()
    hi = np.searchsorted(d, hi_t, side="left")  # days strictly before t_start - lag
    lo = np.searchsorted(d, lo_t, side="left")
    f = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax, "min": np.nanmin, "std": np.nanstd,
         "last": lambda x: x[~np.isnan(x)][-1] if np.any(~np.isnan(x)) else np.nan}[agg]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(ts)):
            if hi[i] > lo[i]:
                out[i] = f(v[lo[i]:hi[i]])
    return pd.DataFrame({"t_start": ts, name or f"x_{series.name or 'series'}_{agg}{window_days}d": out})


def align_deterministic_series(series: pd.Series, panel: pd.DataFrame | None = None, horizon_days: int | None = None,
                               agg: str = "mean", name: str | None = None) -> pd.DataFrame:
    """For series that are KNOWN IN ADVANCE (astronomical tides, ephemerides, calendar effects) it is legitimate to
    summarize the forecast window itself: days D in [t_start, t_start + horizon). NEVER use this for measured data
    (geomagnetic, GNSS, weather, etc.) -- use align_daily_series for those."""
    panel = get_panel() if panel is None else panel
    if horizon_days is None:
        ts_ = np.sort(panel.t_start.unique())
        horizon_days = int((ts_[1] - ts_[0]) / np.timedelta64(1, "D"))
    shifted = series.copy()
    shifted.index = pd.to_datetime(shifted.index) - pd.Timedelta(days=horizon_days)
    # window [t - h, t) on the shifted index == [t, t + h) on the original index
    return align_daily_series(shifted, panel, window_days=horizon_days, agg=agg, lag_days=-horizon_days + horizon_days,
                              name=name or f"x_{series.name or 'det'}_{agg}_fwd{horizon_days}d")


def past_event_count(events: pd.DataFrame, panel: pd.DataFrame | None = None, window_days: float = 365.0,
                     nbr: int = 0, name: str = "x_count", bin_deg: float = 1.0) -> pd.DataFrame:
    """Per-bin count of `events` (needs columns time, latitude) with time in [t_start - window_days, t_start)
    in bins within +-nbr of the panel bin. Strictly causal. Returns [bin_id, t_start, name]."""
    panel = get_panel() if panel is None else panel
    ts = pd.DatetimeIndex(np.sort(panel.t_start.unique()))
    ti = to_days(ts)
    nb = panel.bin_id.nunique()
    eb = lat_to_bin(events.latitude.to_numpy(), bin_deg)
    et = to_days(events.time)
    out = np.zeros((len(ti), nb))
    for b in range(nb):
        tb = np.sort(et[(np.abs(eb - b) <= nbr) & (eb >= 0)])
        out[:, b] = np.searchsorted(tb, ti, side="left") - np.searchsorted(tb, ti - window_days, side="left")
    T, B = np.meshgrid(np.arange(len(ti)), np.arange(nb), indexing="ij")
    return pd.DataFrame({"bin_id": B.ravel(), "t_start": ts[T.ravel()], name: out.ravel()})


def summarize(res: dict, sur: dict | None = None) -> str:
    s = (f"[{res['target']}/{res['model']}] N_test={res['n_test_events']}  IG(full-base)={res['ig_per_event']:+.4f} "
         f"nats/eq  IG(base-Poisson)={res['ig_base_vs_poisson']:+.3f}  AUC base/full={res['base']['roc_auc']:.3f}/"
         f"{res['full']['roc_auc']:.3f}  ASS base/full={res['base']['molchan_ass']:.3f}/{res['full']['molchan_ass']:.3f}"
         f"  T={res['ttest_full_vs_base']['T']:.2f}")
    if sur:
        s += f"  surrogate p={sur['p_value']:.4f} (Bonferroni sig={sur['significant_bonferroni']})"
    return s


if __name__ == "__main__":
    for tm in (6.0, 5.5):
        pnl = build_panel(target_mag=tm, use_cache=True, verbose=True)
        for tg in ("y_all", "y_main"):
            print(tm, summarize(evaluate(None, tg, "logit", pnl)))
