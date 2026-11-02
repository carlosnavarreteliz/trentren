"""Causal per-bin features: RTL, Z-value quiescence, Mogi doughnut, along-strike migration,
inter-bin synchronization / centrality, and long-term stress shadow / loading of past M>=7 ruptures.

All features at issue time t use only events with time < t. Normalizations are expanding past-only.
"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np
import pandas as pd
import harness as H

YR = 365.25
KM_DEG = 111.0
NB = 29

DEFAULTS = dict(
    rtl_r0=100.0, rtl_t0=1.0,          # km, yr
    zq_w=1.0, zq_bg=10.0,              # yr
    dn_w=5.0, dn_lt=15.0,              # yr
    mg_w=365.0, mg_rad=4.0, mg_min=5,  # days, deg, events
    sy_w=5.0, sy_dt=28.0,              # yr, days per interval
    sh_tau=10.0, sh_load_km=150.0, sh_mmin=7.0,
    burn_in_steps=260,                 # 5 yr of weekly issue times before standardization is defined
    depth_max=70.0, min_mag=4.7,
)


def feature_events(min_mag=4.7, depth_max=70.0):
    ev = H.get_events(min_mag=min(min_mag, 4.7))
    ev = ev[ev.mag >= min_mag]
    bg = ev[(ev.bg) & (ev.depth <= depth_max)].sort_values("time").reset_index(drop=True)
    big = ev[(ev.mag >= 7.0) & (ev.depth <= depth_max)].sort_values("time").reset_index(drop=True)
    return bg, big


def _expanding_z(X, burn):
    """X: [T, B]. z-score of X[t] using mean/std of X[0..t-1] (past issue times only). NaN before burn."""
    T = X.shape[0]
    c1 = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])[:-1]
    c2 = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X ** 2, 0)])[:-1]
    n = np.arange(T)[:, None].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = c1 / n
        sd = np.sqrt(np.maximum(c2 / n - mu ** 2, 0))
        z = (X - mu) / np.where(sd > 0, sd, np.nan)
    z[:burn] = np.nan
    return np.where(np.isfinite(z), z, np.nan)


def compute(bg: pd.DataFrame, big: pd.DataFrame, ts: pd.DatetimeIndex, p: dict | None = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(p or {})}
    ti = H.to_days(ts).astype(float)
    T = len(ti)
    centers = H.bin_centers()
    te = H.to_days(bg.time).astype(float)
    lat = bg.latitude.to_numpy()
    mag = bg.mag.to_numpy()
    eb = H.lat_to_bin(lat)
    hist0 = float(H.to_days(pd.DatetimeIndex([H.HISTORY_START]))[0])
    out = {}

    # ---------------- RTL ----------------
    r0, t0 = p["rtl_r0"], p["rtl_t0"] * YR
    Rv, Tv, Lv = (np.zeros((T, NB)) for _ in range(3))
    for b in range(NB):
        d = np.maximum(np.abs(lat - centers[b]) * KM_DEG, 10.0)
        sel = d <= 2 * r0
        if not sel.any():
            continue
        tb, db, mb = te[sel], d[sel], mag[sel]
        li = 10 ** (0.5 * mb - 1.8)
        dt = ti[:, None] - tb[None, :]
        M = (dt > 0) & (dt <= 2 * t0)
        Rv[:, b] = M @ np.exp(-db / r0)
        Tv[:, b] = (M * np.exp(-np.where(M, dt, 0) / t0)).sum(1)
        Lv[:, b] = M @ (li / db)
    burn = p["burn_in_steps"]
    rtl = _expanding_z(Rv, burn) * _expanding_z(Tv, burn) * _expanding_z(Lv, burn)
    out["x_rtl"] = np.cbrt(rtl)

    # helper: cumulative count of events in bin group with time < x
    sorted_by_bin = [np.sort(te[eb == b]) for b in range(NB)]

    def cnt(bins, lo, hi):
        """count of events in the given bins with lo <= time < hi (arrays over T)."""
        s = np.zeros(T)
        for b in bins:
            if 0 <= b < NB:
                tb = sorted_by_bin[b]
                s += np.searchsorted(tb, hi, side="left") - np.searchsorted(tb, lo, side="left")
        return s

    # ---------------- Z-value quiescence ----------------
    W, BGW = p["zq_w"] * YR, p["zq_bg"] * YR
    zq = np.zeros((T, NB))
    for b in range(NB):
        nb_ = [b - 1, b, b + 1]
        nw = cnt(nb_, ti - W, ti)
        lo = np.maximum(ti - W - BGW, hist0)
        nbg = cnt(nb_, lo, ti - W)
        Tbg = np.maximum((ti - W - lo) / YR, 1e-6)
        Tw = W / YR
        lw, lb = nw / Tw, nbg / Tbg
        zq[:, b] = (lb - lw) / np.sqrt(lb / Tbg + lw / Tw + 0.1)
    zq[ti - W - hist0 < 2 * YR] = np.nan   # need >= 2 yr of background
    out["x_zq"] = zq

    # ---------------- Mogi doughnut ----------------
    Wd, LT = p["dn_w"] * YR, p["dn_lt"] * YR
    dn = np.zeros((T, NB))
    for b in range(NB):
        core = [b - 1, b, b + 1]
        ring = [b - 3, b - 2, b + 2, b + 3]
        lo = np.maximum(ti - Wd - LT, hist0)
        s = Wd / np.maximum(ti - Wd - lo, 1.0)
        c5, r5 = cnt(core, ti - Wd, ti), cnt(ring, ti - Wd, ti)
        cl, rl = cnt(core, lo, ti - Wd) * s, cnt(ring, lo, ti - Wd) * s
        dn[:, b] = np.log((r5 + 1) / (c5 + 1)) - np.log((rl + 1) / (cl + 1))
    dn[ti - Wd - hist0 < 3 * YR] = np.nan
    out["x_doughnut"] = dn

    # ---------------- migration ----------------
    Wm, rad, nmin = p["mg_w"], p["mg_rad"], p["mg_min"]
    mg = np.zeros((T, NB))
    for b in range(NB):
        dl = np.abs(lat - centers[b])
        sel = dl <= rad
        tb, db = te[sel] / YR, dl[sel]
        dt = ti[:, None] - te[sel][None, :]
        M = ((dt > 0) & (dt <= Wm)).astype(float)
        n = M.sum(1)
        st, sd = M @ tb, M @ db
        stt, std_ = M @ (tb ** 2), M @ (tb * db)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = stt - st ** 2 / n
            cov = std_ - st * sd / n
            slope = cov / var
        ok = (n >= nmin) & (var > 1e-9)
        mg[:, b] = np.where(ok, -slope, 0.0)
    out["x_migr"] = mg

    # ---------------- synchronization / centrality ----------------
    dtI = p["sy_dt"]
    K = int(round(p["sy_w"] * YR / dtI))
    edges = ti[:, None] - dtI * np.arange(K, -1, -1)[None, :]     # [T, K+1], increasing, last = t
    C = np.zeros((T, K + 1, NB))
    for b in range(NB):
        C[:, :, b] = np.searchsorted(sorted_by_bin[b], edges, side="left")
    X = np.log1p(np.diff(C, axis=1))                               # [T, K, NB]
    Xc = X - X.mean(1, keepdims=True)
    sd = np.sqrt((Xc ** 2).sum(1))                                 # [T, NB]
    Zn = Xc / np.where(sd > 0, sd, np.inf)[:, None, :]
    R = np.einsum("tkb,tkc->tbc", Zn, Zn)                          # [T, NB, NB]
    idx = np.arange(NB)
    D = np.abs(idx[:, None] - idx[None, :])
    nbrmask = ((D >= 1) & (D <= 3)).astype(float)
    sync = (R * nbrmask[None]).sum(2) / nbrmask.sum(1)[None]
    A = np.maximum(R, 0)
    A[:, idx, idx] = 0
    cen = np.zeros((T, NB))
    for t in range(T):
        if A[t].sum() <= 0:
            continue
        w, v = np.linalg.eigh(A[t])
        e = np.abs(v[:, -1])
        cen[t] = e / e.max() if e.max() > 0 else 0
    early = edges[:, 0] < hist0
    sync[early] = np.nan
    cen[early] = np.nan
    out["x_sync"] = sync
    out["x_central"] = cen

    # ---------------- stress shadow / loading ----------------
    tbg = H.to_days(big.time).astype(float)
    mbig = big.mag.to_numpy()
    lbig = big.latitude.to_numpy()
    Lh = 0.5 * 10 ** (-2.44 + 0.59 * mbig)                         # km
    shadow = np.zeros((T, NB))
    load = np.zeros((T, NB))
    for k in range(len(tbg)):
        age = (ti - tbg[k]) / YR
        w = np.where(age > 1.0, 10 ** (0.75 * (mbig[k] - 7.0)) * np.exp(-np.maximum(age, 0) / p["sh_tau"]), 0.0)
        d = np.abs(centers - lbig[k]) * KM_DEG
        inside = (d < Lh[k]).astype(float)
        beyond = ((d >= Lh[k]) & (d < Lh[k] + p["sh_load_km"])).astype(float)
        shadow += w[:, None] * inside[None, :]
        load += w[:, None] * beyond[None, :]
    out["x_shadow"] = np.log1p(shadow)
    out["x_load"] = np.log1p(load)

    Tg, Bg = np.meshgrid(np.arange(T), np.arange(NB), indexing="ij")
    df = pd.DataFrame({"bin_id": Bg.ravel().astype(np.int16), "t_start": ts[Tg.ravel()]})
    for k, v in out.items():
        df[k] = np.asarray(v, float).ravel()
    return df


FAMILIES = {
    "rtl": ["x_rtl"], "zq": ["x_zq"], "doughnut": ["x_doughnut"], "migr": ["x_migr"],
    "sync": ["x_sync", "x_central"], "shadow": ["x_shadow", "x_load"],
}
ALL_COLS = [c for v in FAMILIES.values() for c in v]


def build(panel, p=None, min_mag=4.7):
    bg, big = feature_events(min_mag=min_mag)
    ts = pd.DatetimeIndex(np.sort(panel.t_start.unique()))
    return compute(bg, big, ts, p)
