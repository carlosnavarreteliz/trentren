"""Shared helpers for the tides hypothesis (phase lookup from precomputed tidal-stress extrema)."""
import numpy as np

NB = 29


class Tides:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.lons, self.lats = z["lons"], z["lats"]
        self.ext = {}
        for comp in ("total", "solid", "ocean"):
            for b in range(NB):
                self.ext[(comp, b)] = (z[f"ext_t_{comp}_{b}"], z[f"ext_max_{comp}_{b}"])
        self.day0 = float(z["day0"])
        self.dmax = np.vstack([z[f"dmax_{b}"] for b in range(NB)])   # (bin, day) Pa
        self.dmin = np.vstack([z[f"dmin_{b}"] for b in range(NB)])
        t = self.ext[("total", 0)][0]
        self.tmin, self.tmax = float(t[2]), float(t[-3])

    def phase(self, bins, t_days, comp="total"):
        """Tanaka phase (radians): 0 at dCFS max, +-pi at min; positive max->min, negative min->max."""
        bins = np.asarray(bins); t = np.asarray(t_days, float)
        out = np.full(len(t), np.nan)
        for b in np.unique(bins):
            m = bins == b
            te, im = self.ext[(comp, int(b))]
            k = np.searchsorted(te, t[m], side="right") - 1
            k = np.clip(k, 0, len(te) - 2)
            frac = (t[m] - te[k]) / (te[k + 1] - te[k])
            frac = np.clip(frac, 0, 1)
            out[m] = np.where(im[k], np.pi * frac, -np.pi + np.pi * frac)
        return out

    def wrap(self, t):
        """wrap times into the valid grid range (for tide-shift surrogates)."""
        L = self.tmax - self.tmin
        return self.tmin + np.mod(np.asarray(t) - self.tmin, L)
