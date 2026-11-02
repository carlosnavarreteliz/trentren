"""Causal per-bin tidal-sensitivity features aligned to a harness panel."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import harness as H

NB = 29


class FeatureBuilder:
    def __init__(self, tides, ev_phases, panel, bg_seed=777, n_bg=20000):
        """ev_phases: event_phases.parquet frame (all depth<=70 events, M>=4.7)."""
        self.T = tides
        self.panel = panel
        self.ts = np.sort(panel.t_start.unique())
        self.ti = H.to_days(pd.DatetimeIndex(self.ts))
        # background moments per bin from random times 1973-2012 (train period only)
        rng = np.random.default_rng(bg_seed)
        lo, hi = H.to_days(pd.Timestamp("1973-01-01")), H.to_days(H.TEST_START)
        self.c0 = np.zeros(NB); self.v0 = np.ones(NB); self.m0 = np.zeros(NB, complex)
        for b in range(NB):
            ph = tides.phase(np.full(n_bg // 4, b), rng.uniform(lo, hi, n_bg // 4))
            self.c0[b], self.v0[b] = np.cos(ph).mean(), np.cos(ph).var()
            self.m0[b] = np.exp(1j * ph).mean()
        self.ev = ev_phases

    def event_set(self, min_mag=4.7, bg_only=True):
        e = self.ev[(self.ev.mag >= min_mag)]
        if bg_only:
            e = e[e.bg]
        return e.sort_values("t_days").reset_index(drop=True)

    def build(self, e, window=1095.0, shift=0.0, min_n=5, nbr=1, prefix="x_"):
        """Rolling tidal-sensitivity features using events with t in [t_start - window, t_start).
        shift != 0: phases evaluated at (t + shift) wrapped in the grid (tide-shift surrogate)."""
        t = e.t_days.to_numpy(); b = e.bin_id.to_numpy()
        if shift == 0:
            ph = e.ph_total.to_numpy()
        else:
            ph = self.T.phase(b, self.T.wrap(t + shift))
        xc = (np.cos(ph) - self.c0[b]) / np.sqrt(self.v0[b])
        zc = np.exp(1j * ph) - self.m0[b]
        nT = len(self.ti)
        Z = np.full((nT, NB), np.nan); LP = np.full((nT, NB), np.nan); N = np.zeros((nT, NB))
        for bb in range(NB):
            m = (b >= bb - nbr) & (b <= bb + nbr)
            tb = t[m]
            cx = np.concatenate([[0], np.cumsum(xc[m])])
            cz = np.concatenate([[0], np.cumsum(zc[m])])
            hi = np.searchsorted(tb, self.ti, side="left")
            lo = np.searchsorted(tb, self.ti - window, side="left")
            n = hi - lo
            N[:, bb] = n
            ok = n >= min_n
            Z[ok, bb] = (cx[hi] - cx[lo])[ok] / np.sqrt(n[ok])
            R2 = np.abs(cz[hi] - cz[lo]) ** 2
            LP[ok, bb] = (R2[ok] / n[ok]) / np.log(10)
        TT, BB = np.meshgrid(np.arange(nT), np.arange(NB), indexing="ij")
        return pd.DataFrame({"bin_id": BB.ravel(), "t_start": self.ts[TT.ravel()],
                             f"{prefix}tsens": Z.ravel(), f"{prefix}tsens_lp": LP.ravel(),
                             f"{prefix}tsens_n": N.ravel()})

    def amplitude(self, horizon_days):
        """Deterministic: max total dCFS (kPa) at the bin point over [t_start, t_start + horizon)."""
        d0 = (np.floor(self.ti) - self.T.day0).astype(int)
        nT = len(self.ti)
        A = np.zeros((nT, NB))
        for k in range(int(horizon_days)):
            A = np.maximum(A, self.T.dmax[:, d0 + k].T) if k else self.T.dmax[:, d0].T.copy()
        TT, BB = np.meshgrid(np.arange(nT), np.arange(NB), indexing="ij")
        return pd.DataFrame({"bin_id": BB.ravel(), "t_start": self.ts[TT.ravel()], "x_tamp": A.ravel() / 1e3})
