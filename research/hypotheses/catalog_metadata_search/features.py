"""
Causal catalog-derived feature library for the catalog_metadata_search hypothesis.

Every feature at issue time t uses ONLY events with time < t (window [t - W, t)).
Features are computed on the weekly grid of the harness panel (1976-01-01 + 7k) and can be aligned
"as of" (backward) to any other panel grid.

Usage (as a library):  F = build_feature_library(raw_csv_path)   -> DataFrame [bin_id, t_start, x_*]
"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np
import pandas as pd
import harness as H

NB = 29
LOG10E = np.log10(np.e)


# ---------------------------------------------------------------------------------------------
# event tables
# ---------------------------------------------------------------------------------------------
def load_events(raw_csv):
    """Set A: M>=4.7 events (harness, with causal ZBZ `bg` flag) merged with ComCat metadata.
    Set S: all magnitudes (incomplete before ~1990; detection-dominated), with metadata."""
    raw = pd.read_csv(raw_csv, low_memory=False)
    meta_cols = ["id", "nst", "gap", "dmin", "rms", "horizontalError", "depthError", "magError", "magNst",
                 "locationSource", "magSource"]
    meta = raw[meta_cols].drop_duplicates("id")
    A = H.get_events(min_mag=H.M_MODEL)            # includes bg (causal) and mainshock (NOT used)
    A = A.drop(columns=["mainshock", "cluster_id"]).merge(meta, on="id", how="left")
    S = H.load_catalog(min_mag=None).merge(meta, on="id", how="left")
    S = S[S.time >= H.HISTORY_START].reset_index(drop=True)
    for E in (A, S):
        hr = (E.time.dt.hour + E.time.dt.minute / 60.0).to_numpy()
        E["hcos"] = np.cos(2 * np.pi * hr / 24.0)
        E["hsin"] = np.sin(2 * np.pi * hr / 24.0)
        # local night in Chile (~UTC-4): 22:00-06:00 local = 02:00-10:00 UTC
        E["night"] = ((hr >= 2) & (hr < 10)).astype(float)
        E["weekend"] = (E.time.dt.dayofweek >= 5).astype(float)
        mt = E.magType.fillna("").str.lower()
        E["is_mw"] = mt.str.startswith("mw").astype(float)
        E["is_mb"] = (mt == "mb").astype(float)
        E["is_ml"] = mt.isin(["ml", "md", "m"]).astype(float)
        E["is_md"] = (mt == "md").astype(float)
        E["is_guc"] = E.locationSource.fillna("").str.contains("guc").astype(float)
        E["fixdep"] = np.isin(np.round(E.depth.to_numpy(), 2), [10.0, 33.0, 35.0]).astype(float)
        E["nst_na"] = E.nst.isna().astype(float)
    return A.sort_values("time").reset_index(drop=True), S.sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# prefix-sum machinery
# ---------------------------------------------------------------------------------------------
def _csum(q):
    """prefix sums with NaN -> 0, plus prefix counts of finite values. Returns (C, Cn) length n+1."""
    q = np.asarray(q, float)
    ok = np.isfinite(q)
    C = np.concatenate([[0.0], np.cumsum(np.where(ok, q, 0.0))])
    Cn = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    return C, Cn


class Window:
    """events of one spatial neighbourhood; window sums over [t-W, t) for a grid of issue times."""

    def __init__(self, E, ti):
        self.E = E
        self.t = E.t_days.to_numpy()
        self.ti = ti
        self.cache = {}

    def idx(self, W):
        if W not in self.cache:
            hi = np.searchsorted(self.t, self.ti, side="left")
            lo = np.searchsorted(self.t, self.ti - W, side="left")
            self.cache[W] = (lo, hi)
        return self.cache[W]

    def n(self, W):
        lo, hi = self.idx(W)
        return (hi - lo).astype(float)

    def mean(self, q, W, min_n=5):
        C, Cn = _csum(q)
        lo, hi = self.idx(W)
        s, k = C[hi] - C[lo], Cn[hi] - Cn[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(k >= min_n, s / k, np.nan)

    def std(self, q, W, min_n=10):
        m1 = self.mean(q, W, min_n)
        m2 = self.mean(np.asarray(q, float) ** 2, W, min_n)
        return np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))

    def dt_stats(self, W, min_n=6):
        """inter-event times (days) for consecutive pairs inside the window."""
        dt = np.diff(self.t, prepend=np.nan)
        ldt = np.log10(dt + 1e-4)
        lo, hi = self.idx(W)
        out = {}
        for nm, q in (("dt", dt), ("dt2", dt ** 2), ("ldt", ldt), ("ldt2", ldt ** 2)):
            C, Cn = _csum(q)
            lo1 = np.minimum(lo + 1, hi)
            out[nm] = C[hi] - C[lo1]
            out["k"] = Cn[hi] - Cn[lo1]
        k = out["k"]
        ok = k >= min_n - 1
        with np.errstate(invalid="ignore", divide="ignore"):
            mdt = out["dt"] / k
            sdt = np.sqrt(np.maximum(out["dt2"] / k - mdt ** 2, 0))
            ml = out["ldt"] / k
            sl = np.sqrt(np.maximum(out["ldt2"] / k - ml ** 2, 0))
        return (np.where(ok, sdt / mdt, np.nan), np.where(ok, ml, np.nan), np.where(ok, sl, np.nan))

    def mag_entropy(self, W, min_n=10):
        m = self.E.mag.to_numpy()
        edges = [4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9, 6.1, 99]
        lo, hi = self.idx(W)
        P = []
        for a, b in zip(edges[:-1], edges[1:]):
            C, _ = _csum(((m >= a) & (m < b)).astype(float))
            P.append(C[hi] - C[lo])
        P = np.array(P)
        n = P.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = P / n
            ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=0)
        return np.where(n >= min_n, ent, np.nan)

    def d2(self, W, min_n=10, r1=25.0, r2=100.0):
        """correlation-dimension proxy of epicentres: log(C(r1)/C(r2))/log(r1/r2)."""
        la, lo_ = self.E.latitude.to_numpy(), self.E.longitude.to_numpy()
        lo, hi = self.idx(W)
        out = np.full(len(self.ti), np.nan)
        last = (-1, -1)
        val = np.nan
        for i in range(len(self.ti)):
            a, b = lo[i], hi[i]
            if (a, b) != last:
                last = (a, b)
                if b - a >= min_n:
                    x, y = la[a:b], lo_[a:b]
                    D = H._haversine_km(x[:, None], y[:, None], x[None, :], y[None, :])
                    iu = np.triu_indices(b - a, 1)
                    d = D[iu]
                    c1, c2 = np.sum(d < r1), np.sum(d < r2)
                    val = np.log(c1 / c2) / np.log(r1 / r2) if (c1 > 0 and c2 > 0) else np.nan
                else:
                    val = np.nan
            out[i] = val
        return out


def weekly_count_dynamics(t_ev, ti):
    """counts per weekly grid step (events in [t_{k-1}, t_k)), then causal rolling statistics."""
    N = np.searchsorted(np.sort(t_ev), ti, side="left").astype(float)
    c = np.diff(N, prepend=N[0])
    cs = pd.Series(c)
    m52 = cs.rolling(52, min_periods=52).mean()
    v52 = cs.rolling(52, min_periods=52).var()
    m156 = cs.rolling(156, min_periods=156).mean()
    v156 = cs.rolling(156, min_periods=156).var()
    allan = (cs.diff() ** 2).rolling(52, min_periods=52).mean() / (2 * m52)
    # aggregated-variance Hurst exponent over the last 156 weeks, scales m = 1,2,4,8 weeks
    lv = []
    ms = np.array([1, 2, 4, 8])
    for m in ms:
        sm = pd.Series(N - np.r_[np.full(m, np.nan), N[:-m]])
        lv.append(np.log(sm.rolling(156, min_periods=156).var().to_numpy()))
    lv = np.array(lv)
    xl = np.log(ms) - np.log(ms).mean()
    with np.errstate(invalid="ignore"):
        slope = (xl[:, None] * (lv - lv.mean(0))).sum(0) / (xl ** 2).sum()
    hurst = np.where(np.all(np.isfinite(lv), axis=0), slope / 2.0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"fano52": (v52 / m52).to_numpy(), "fano156": (v156 / m156).to_numpy(),
                "allan52": allan.to_numpy(), "hurst156": hurst}


def _sub(E, b, nbr):
    if b is None:
        return E
    return E[(E.bin_id >= b - nbr) & (E.bin_id <= b + nbr) & (E.bin_id >= 0)].reset_index(drop=True)


def neighbourhood_features(A, S, ti, b, nbr, full=True):
    """dict name -> array[T] for one neighbourhood (b=None: whole region)."""
    a, s = _sub(A, b, nbr), _sub(S, b, nbr)
    wa, ws = Window(a, ti), Window(s, ti)
    f = {}
    if full:
        for W in (90, 365, 1095):
            mag = a.mag.to_numpy()
            f[f"a_meanmag_{W}"] = wa.mean(mag, W)
            f[f"a_stdmag_{W}"] = wa.std(mag, W)
            mm = wa.mean(mag, W, min_n=10)
            f[f"a_bval_{W}"] = LOG10E / np.maximum(mm - (H.M_MODEL - 0.05), 0.05)
            f[f"a_magent_{W}"] = wa.mag_entropy(W)
            dep = a.depth.to_numpy()
            f[f"a_meandep_{W}"] = wa.mean(dep, W)
            f[f"a_stddep_{W}"] = wa.std(dep, W)
            f[f"a_fshallow_{W}"] = wa.mean((dep < 35).astype(float), W)
            f[f"a_fdeep_{W}"] = wa.mean((dep >= 80).astype(float), W)
            f[f"a_fixdep_{W}"] = wa.mean(a.fixdep.to_numpy(), W)
            f[f"a_meanlon_{W}"] = wa.mean(a.longitude.to_numpy(), W)
            f[f"a_stdlon_{W}"] = wa.std(a.longitude.to_numpy(), W)
            f[f"a_stdlat_{W}"] = wa.std(a.latitude.to_numpy(), W)
            cv, mld, sld = wa.dt_stats(W)
            f[f"a_cvdt_{W}"], f[f"a_meanlogdt_{W}"], f[f"a_stdlogdt_{W}"] = cv, mld, sld
            f[f"a_fbg_{W}"] = wa.mean(a.bg.to_numpy().astype(float), W)
            n = wa.n(W)
            lo, hi = wa.idx(W)
            Cb, _ = _csum(10 ** (0.75 * mag))
            Cm, _ = _csum(10 ** (1.5 * mag))
            f[f"a_logbenioff_{W}"] = np.log10(Cb[hi] - Cb[lo] + 1.0)
            f[f"a_logmoment_{W}"] = np.log10(Cm[hi] - Cm[lo] + 1.0)
            C1 = wa.mean(a.hcos.to_numpy(), W)
            S1 = wa.mean(a.hsin.to_numpy(), W)
            f[f"a_hourR_{W}"] = np.sqrt(C1 ** 2 + S1 ** 2)
            f[f"a_night_{W}"] = wa.mean(a.night.to_numpy(), W)
            f[f"a_weekend_{W}"] = wa.mean(a.weekend.to_numpy(), W)
            f[f"a_fmw_{W}"] = wa.mean(a.is_mw.to_numpy(), W)
            f[f"a_fmb_{W}"] = wa.mean(a.is_mb.to_numpy(), W)
            f[f"a_fml_{W}"] = wa.mean(a.is_ml.to_numpy(), W)
            # ---- ComCat metadata (network-dependent, partly missing before 2000) ----
            f[f"m_rms_{W}"] = wa.mean(a.rms.to_numpy(float), W, min_n=3)
            f[f"m_nst_{W}"] = wa.mean(a.nst.to_numpy(float), W, min_n=3)
            f[f"m_gap_{W}"] = wa.mean(a.gap.to_numpy(float), W, min_n=3)
            f[f"m_dmin_{W}"] = wa.mean(a.dmin.to_numpy(float), W, min_n=3)
            f[f"m_herr_{W}"] = wa.mean(a.horizontalError.to_numpy(float), W, min_n=3)
            f[f"m_derr_{W}"] = wa.mean(a.depthError.to_numpy(float), W, min_n=3)
            f[f"m_merr_{W}"] = wa.mean(a.magError.to_numpy(float), W, min_n=3)
            f[f"m_magnst_{W}"] = wa.mean(a.magNst.to_numpy(float), W, min_n=3)
            f[f"m_fguc_{W}"] = wa.mean(a.is_guc.to_numpy(), W)
            f[f"m_nstna_{W}"] = wa.mean(a.nst_na.to_numpy(), W)
            # ---- all-magnitude set S (detection-dominated) ----
            f[f"s_logcnt_{W}"] = np.log1p(ws.n(W))
            f[f"s_ratiosmall_{W}"] = np.log((ws.n(W) + 1.0) / (n + 1.0))
            f[f"s_meanmag_{W}"] = ws.mean(s.mag.to_numpy(), W)
            f[f"s_rms_{W}"] = ws.mean(s.rms.to_numpy(float), W, min_n=3)
            f[f"s_fmd_{W}"] = ws.mean(s.is_md.to_numpy(), W)
            f[f"s_fguc_{W}"] = ws.mean(s.is_guc.to_numpy(), W)
            f[f"s_night_{W}"] = ws.mean(s.night.to_numpy(), W)
            if W >= 365:
                cvs, _, _ = ws.dt_stats(W)
                f[f"s_cvdt_{W}"] = cvs
        f["a_logcnt_90"] = np.log1p(wa.n(90))
        f["a_logcnt_1095"] = np.log1p(wa.n(1095))
        f["s_logcnt_30"] = np.log1p(ws.n(30))
        f["a_meanmag_30"] = wa.mean(a.mag.to_numpy(), 30, min_n=3)
        f["a_meandep_30"] = wa.mean(a.depth.to_numpy(), 30, min_n=3)
        # temporal contrasts (recent 90 d vs 1095 d)
        for k in ("meanmag", "meandep", "bval", "meanlon", "fbg"):
            f[f"c_{k}_90v1095"] = f[f"a_{k}_90"] - f[f"a_{k}_1095"]
        f["c_rms_90v1095"] = f["m_rms_90"] - f["m_rms_1095"]
        f["c_zrate_90v1095"] = np.log((wa.n(90) + 0.5) / 90.0) - np.log((wa.n(1095) + 0.5) / 1095.0)
        for k, v in weekly_count_dynamics(a.t_days.to_numpy(), ti).items():
            f[f"w_a_{k}"] = v
        for k, v in weekly_count_dynamics(s.t_days.to_numpy(), ti).items():
            if k in ("fano52", "hurst156"):
                f[f"w_s_{k}"] = v
        f["sp_d2_1095"] = wa.d2(1095)
    else:
        # reduced set for the wide (+-3 bins) neighbourhood and the whole region
        mag = a.mag.to_numpy()
        f["a_logcnt_365"] = np.log1p(wa.n(365))
        f["a_logcnt_1095"] = np.log1p(wa.n(1095))
        for W in (365, 1095):
            mm = wa.mean(mag, W, min_n=10)
            f[f"a_bval_{W}"] = LOG10E / np.maximum(mm - (H.M_MODEL - 0.05), 0.05)
            f[f"a_cvdt_{W}"] = wa.dt_stats(W)[0]
            f[f"a_fbg_{W}"] = wa.mean(a.bg.to_numpy().astype(float), W)
            f[f"a_meandep_{W}"] = wa.mean(a.depth.to_numpy(), W)
        f["a_night_365"] = wa.mean(a.night.to_numpy(), 365)
        f["a_fmw_365"] = wa.mean(a.is_mw.to_numpy(), 365)
        f["m_rms_365"] = wa.mean(a.rms.to_numpy(float), 365, min_n=3)
        f["s_ratiosmall_365"] = np.log((ws.n(365) + 1.0) / (wa.n(365) + 1.0))
        f["s_logcnt_365"] = np.log1p(ws.n(365))
        dyn = weekly_count_dynamics(a.t_days.to_numpy(), ti)
        f["w_a_fano52"], f["w_a_hurst156"] = dyn["fano52"], dyn["hurst156"]
        f["sp_d2_1095"] = wa.d2(1095)
    return f


def build_feature_library(raw_csv, grid=None, verbose=True):
    A, S = load_events(raw_csv)
    if grid is None:
        grid = pd.DatetimeIndex(np.sort(H.get_panel(7, 7, 6.0).t_start.unique()))
    ti = H.to_days(grid)
    T = len(ti)
    cols = {}
    for b in range(NB):
        f1 = neighbourhood_features(A, S, ti, b, 1, full=True)
        f3 = neighbourhood_features(A, S, ti, b, 3, full=False)
        for k, v in f1.items():
            cols.setdefault(f"x_n1_{k}", np.full((T, NB), np.nan))[:, b] = v
        for k, v in f3.items():
            cols.setdefault(f"x_n3_{k}", np.full((T, NB), np.nan))[:, b] = v
        if verbose and b % 7 == 0:
            print(f"  bin {b} done, {len(cols)} features", flush=True)
    fg = neighbourhood_features(A, S, ti, None, 0, full=False)
    for k, v in fg.items():
        cols[f"x_gl_{k}"] = np.repeat(np.asarray(v)[:, None], NB, axis=1)
    TT, BB = np.meshgrid(np.arange(T), np.arange(NB), indexing="ij")
    out = pd.DataFrame({"bin_id": BB.ravel().astype(np.int16), "t_start": grid[TT.ravel()]})
    for k, v in cols.items():
        out[k] = v.ravel().astype(np.float32)
    return out


def align_asof(F, panel):
    """align a weekly-grid feature frame to another panel's issue times, backward as-of (causal)."""
    tp = pd.DatetimeIndex(np.sort(panel.t_start.unique()))
    tg = pd.DatetimeIndex(np.sort(F.t_start.unique()))
    j = np.searchsorted(tg.values, tp.values, side="right") - 1  # last grid point <= t_start
    ok = j >= 0
    m = pd.DataFrame({"t_start": tp[ok], "t_grid": tg[j[ok]]})
    out = m.merge(F.rename(columns={"t_start": "t_grid"}), on="t_grid").drop(columns="t_grid")
    return out
