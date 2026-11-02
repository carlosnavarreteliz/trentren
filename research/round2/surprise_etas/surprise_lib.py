"""Library for round2/surprise_etas: 1-D (latitude) x time ETAS fit on training data and causal surprise features.

All features at issue time t use only events with t_j < t. ETAS parameters, b-value, background KDE, depth KDEs and
the empirical CDFs used by the composite are estimated on the training period only (< 2013-01-01).
"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import minimize
import torch
import harness as H

torch.set_num_threads(2)
LAT0, LAT1 = H.LAT_MIN, H.LAT_MAX
TRAIN_END = H.to_days(H.TEST_START).item()
FIT_START = H.to_days(H.PANEL_START).item()
NB = 29
EDGES = LAT0 + np.arange(NB + 1) * 1.0
LN10 = np.log(10.0)
SQ2PI = np.sqrt(2 * np.pi)


# --------------------------------------------------------------------------------------------- background KDE
def bg_kde(ev, bw=0.5):
    """Normalized (over [LAT0, LAT1]) Gaussian KDE of training-period latitudes of causal-background events."""
    tr = ev[(ev.t_days < TRAIN_END) & ev.bg]
    y = tr.latitude.to_numpy()
    norm = np.mean(ndtr((LAT1 - y) / bw) - ndtr((LAT0 - y) / bw))

    def f(q):
        q = np.asarray(q, float)
        out = np.exp(-0.5 * ((q[..., None] - y) / bw) ** 2).sum(-1) / (len(y) * bw * SQ2PI) / norm
        return out

    def mass(lo, hi):
        return np.mean(ndtr((hi - y) / bw) - ndtr((lo - y) / bw)) / norm
    return f, mass


# --------------------------------------------------------------------------------------------- ETAS fit
def unpack(th):
    mu0, K, alpha, c, pm1, d, gam = [torch.exp(x) for x in th]
    return mu0, K, alpha, c, 1 + pm1, d, gam


def fit_etas(ev, m0, verbose=True):
    """MLE of (mu0,K,alpha,c,p,d,gamma) on target events FIT_START <= t < TRAIN_END; history from 1973."""
    ev = ev[ev.t_days < TRAIN_END].reset_index(drop=True)
    t = torch.tensor(ev.t_days.to_numpy())
    y = torch.tensor(ev.latitude.to_numpy())
    m = torch.tensor(ev.mag.to_numpy() - m0)
    fbg, _ = bg_kde(ev)
    fb = torch.tensor(fbg(ev.latitude.to_numpy()))
    tgt = (ev.t_days.to_numpy() >= FIT_START)
    idx_t = np.where(tgt)[0]
    # all pairs (i target, j < i)
    ii, jj = np.tril_indices(len(ev), -1)
    keep = tgt[ii]
    ii, jj = torch.tensor(ii[keep]), torch.tensor(jj[keep])
    dt = t[ii] - t[jj]
    dy = y[ii] - y[jj]
    mj = m[jj]
    T0, T1 = FIT_START, TRAIN_END
    tidx = torch.tensor(idx_t)
    if verbose:
        print(f"fit: {len(ev)} events, {len(idx_t)} targets, {len(ii)/1e6:.1f}M pairs", flush=True)

    def nll(th):
        mu0, K, alpha, c, p, d, gam = unpack(th)
        sig = d * torch.exp(0.5 * gam * LN10 * mj)
        trig = K * torch.exp(alpha * mj) * (p - 1) * c ** (p - 1) * (dt + c) ** (-p) \
            * torch.exp(-0.5 * (dy / sig) ** 2) / (sig * SQ2PI)
        lam = mu0 * fb[tidx] + torch.zeros(len(ev), dtype=torch.float64).index_add(0, ii, trig)[tidx]
        # integral
        sigall = d * torch.exp(0.5 * gam * LN10 * m)
        smass = torch.special.ndtr((LAT1 - y) / sigall) - torch.special.ndtr((LAT0 - y) / sigall)
        a = torch.clamp(T0 - t, min=0.0)
        b = T1 - t
        G = lambda x: 1 - (1 + x / c) ** (1 - p)
        integ = mu0 * (T1 - T0) + (K * torch.exp(alpha * m) * (G(b) - G(a)) * smass).sum()
        return -(torch.log(lam).sum() - integ)

    def fg(x):
        th = torch.tensor(x, requires_grad=True)
        v = nll(th)
        v.backward()
        return v.item(), th.grad.numpy().copy()

    rate = len(idx_t) / (T1 - T0)
    x0 = np.log([0.5 * rate, 0.5, 1.5, 0.02, 0.1, 0.3, 0.5])
    best = None
    for start in (x0, np.log([0.6 * rate, 0.3, 2.0, 0.05, 0.2, 0.2, 1.0])):
        r = minimize(fg, start, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        if verbose:
            print("  start ->", r.fun, r.message, flush=True)
        if best is None or r.fun < best.fun:
            best = r
    names = ["mu0", "K", "alpha", "c", "p", "d", "gamma"]
    vals = [v.item() for v in unpack(torch.tensor(best.x))]
    par = dict(zip(names, vals))
    par["nll"] = float(best.fun)
    par["m0"] = m0
    # branching ratio n = K (kernels normalized; alpha*... magnitude productivity averaged over GR)
    mags = ev.mag.to_numpy()[tgt] - m0
    par["branching"] = float(par["K"] * np.mean(np.exp(par["alpha"] * mags)))
    if verbose:
        print("  params:", {k: round(v, 5) if isinstance(v, float) else v for k, v in par.items()}, flush=True)
    return par


# --------------------------------------------------------------------------------------------- model pieces
class ETAS:
    def __init__(self, ev, par):
        self.ev = ev.reset_index(drop=True)
        self.par = par
        self.m0 = par["m0"]
        self.fbg, self.bgmass = bg_kde(self.ev)
        self.t = self.ev.t_days.to_numpy()
        self.y = self.ev.latitude.to_numpy()
        self.m = self.ev.mag.to_numpy() - self.m0
        p = par
        self.Kj = p["K"] * np.exp(p["alpha"] * self.m)
        self.sig = p["d"] * np.exp(0.5 * p["gamma"] * LN10 * self.m)
        self.Fb = np.array([self.bgmass(EDGES[b], EDGES[b + 1]) for b in range(NB)])
        # S_jb: spatial kernel mass of parent j in bin b
        self.S = ndtr((EDGES[None, 1:] - self.y[:, None]) / self.sig[:, None]) - \
            ndtr((EDGES[None, :-1] - self.y[:, None]) / self.sig[:, None])

    def g(self, dt):
        p, c = self.par["p"], self.par["c"]
        return (p - 1) * c ** (p - 1) * (dt + c) ** (-p)

    def G(self, x):
        p, c = self.par["p"], self.par["c"]
        return 1 - (1 + x / c) ** (1 - p)

    def compensator_bins(self, T, chunk=600):
        """C_b(T) = int_{t_ref}^{T} int_bin lambda, for arrays T (days); uses only events with t_j < T."""
        T = np.asarray(T, float)
        t_ref = self.t[0]
        out = np.empty((len(T), NB))
        for s in range(0, len(T), chunk):
            tt = T[s:s + chunk]
            dt = tt[:, None] - self.t[None, :]
            W = np.where(dt > 0, self.Kj[None, :] * self.G(np.maximum(dt, 0.0)), 0.0)
            out[s:s + chunk] = W @ self.S + self.par["mu0"] * (tt - t_ref)[:, None] * self.Fb[None, :]
        return out

    def event_intensity_and_spatial(self, grid_step=0.05, chunk=400):
        """For each event i: lambda(t_i, y_i) (per day per degree), spatial surprise s_i - H_i of its latitude under
        the normalized ETAS spatial density at t_i (parents strictly earlier)."""
        n = len(self.t)
        grid = np.arange(LAT0 + grid_step / 2, LAT1, grid_step)
        fbg_grid = self.fbg(grid)
        fbg_ev = self.fbg(self.y)
        Phi = np.exp(-0.5 * ((grid[None, :] - self.y[:, None]) / self.sig[:, None]) ** 2) / (self.sig[:, None] * SQ2PI)
        lam_ev = np.empty(n)
        spat = np.empty(n)
        mu0 = self.par["mu0"]
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            dt = self.t[s:e, None] - self.t[None, :]
            W = np.where(dt > 0, self.Kj[None, :] * self.g(np.maximum(dt, 0.0)), 0.0)
            # at the event's own latitude
            ker = np.exp(-0.5 * ((self.y[s:e, None] - self.y[None, :]) / self.sig[None, :]) ** 2) / (self.sig[None, :] * SQ2PI)
            lam_ev[s:e] = mu0 * fbg_ev[s:e] + (W * ker).sum(1)
            Lg = mu0 * fbg_grid[None, :] + W @ Phi
            Z = Lg.sum(1) * grid_step
            f = Lg / Z[:, None]
            H_ = -(f * np.log(np.maximum(f, 1e-300))).sum(1) * grid_step
            spat[s:e] = -np.log(lam_ev[s:e] / Z) - H_
        return lam_ev, spat


# --------------------------------------------------------------------------------------------- features
def ad_stat(u):
    u = np.clip(np.sort(u), 1e-12, 1 - 1e-12)
    n = len(u)
    i = np.arange(1, n + 1)
    return -n - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1])))


def depth_surprise(ev, bw=10.0):
    """-log p_bin(z_i) - H_bin with training-period depth KDE of the event's own bin (pool +-2 bins if < 30 events)."""
    tr = ev[ev.t_days < TRAIN_END]
    grid = np.arange(0, 720, 1.0)
    out = np.full(len(ev), np.nan)
    for b in range(NB):
        k = 0
        while True:
            z = tr.depth[(tr.bin_id >= b - k) & (tr.bin_id <= b + k)].to_numpy()
            if len(z) >= 30 or k >= 5:
                break
            k += 1
        dens = np.exp(-0.5 * ((grid[:, None] - z[None, :]) / bw) ** 2).sum(1)
        dens = dens / (dens.sum() * 1.0)
        dens = np.maximum(dens, 1e-8)
        dens = dens / dens.sum()
        Hb = -(dens * np.log(dens)).sum()
        sel = (ev.bin_id == b).to_numpy()
        zi = np.clip(np.round(ev.depth.to_numpy()[sel]).astype(int), 0, len(grid) - 1)
        out[sel] = -np.log(dens[zi]) - Hb
    return out


def build_features(ev, par, panel, n_last=20, verbose=True):
    """Per (bin, issue time) causal surprise components. Returns DataFrame [bin_id, t_start, x_*]."""
    ev = ev.reset_index(drop=True)
    model = ETAS(ev, par)
    m0 = par["m0"]
    tr = ev[(ev.t_days < TRAIN_END) & (ev.t_days >= FIT_START)]
    beta = 1.0 / np.mean(tr.mag.to_numpy() - (m0 - 0.05))   # Aki-Utsu, Delta m = 0.1
    if verbose:
        print(f"  b-value (train) = {beta / LN10:.3f}", flush=True)
    lam_ev, spat_ev = model.event_intensity_and_spatial()
    dep_ev = depth_surprise(ev)
    tiss_ts = np.sort(panel.t_start.unique())
    tiss = H.to_days(pd.DatetimeIndex(tiss_ts))
    C_iss = model.compensator_bins(tiss)
    C_ev = model.compensator_bins(model.t)          # at event times (events strictly before)
    rows = []
    tE = model.t
    bE = ev.bin_id.to_numpy()
    mE = ev.mag.to_numpy()
    W = {30: None, 90: None, 365: None}
    C_win = {w: model.compensator_bins(tiss - w) for w in W}
    for b in range(NB):
        nb = [k for k in (b - 1, b, b + 1) if 0 <= k < NB]
        width = float(len(nb))
        inB = np.isin(bE, nb)
        tB = tE[inB]
        lamB = lam_ev[inB]
        spB, dpB, mB = spat_ev[inB], dep_ev[inB], mE[inB]
        CB_ev = C_ev[inB][:, nb].sum(1)
        CB_iss = C_iss[:, nb].sum(1)
        # number of B events strictly before each issue time
        kpos = np.searchsorted(tB, tiss, side="left")
        out = {"bin_id": np.full(len(tiss), b), "t_start": tiss_ts}
        for w in (30, 90, 365):
            lo = np.searchsorted(tB, tiss - w, side="left")
            N = kpos - lo
            Lam = CB_iss - C_win[w][:, nb].sum(1)
            out[f"x_rr{w}"] = np.log((N + 0.5) / (Lam + 0.5))
            cl = np.concatenate([[0.0], np.cumsum(np.log(lamB))])
            sumlog = cl[kpos] - cl[lo]
            Nf = np.maximum(N, 1)
            ll_etas = sumlog - Lam
            ll_pois = N * np.log(Nf / (w * width)) - N
            lld = (ll_etas - ll_pois) / Nf
            out[f"x_lld{w}"] = np.where(N >= 3, lld, np.nan)
        # last-n events statistics
        ad = np.full(len(tiss), np.nan); gap = np.full(len(tiss), np.nan)
        mg = np.full(len(tiss), np.nan); sp = np.full(len(tiss), np.nan); dp = np.full(len(tiss), np.nan)
        for q, k in enumerate(kpos):
            if k >= 1:
                gap[q] = np.log(max(CB_iss[q] - CB_ev[k - 1], 1e-6))
            if k >= n_last + 1:
                dtau = np.diff(CB_ev[k - n_last - 1:k])
                ad[q] = ad_stat(1 - np.exp(-np.maximum(dtau, 1e-12)))
            if k >= n_last:
                sl = slice(k - n_last, k)
                mg[q] = (np.mean(mB[sl] - (m0 - 0.05)) * beta - 1) * np.sqrt(n_last)
                sp[q] = np.mean(spB[sl])
                dp[q] = np.nanmean(dpB[sl])
        out.update({"x_ad": ad, "x_gap": gap, "x_mag": mg, "x_spat": sp, "x_depth": dp})
        rows.append(pd.DataFrame(out))
    F = pd.concat(rows, ignore_index=True).sort_values(["t_start", "bin_id"]).reset_index(drop=True)
    return F, model, beta


COMPONENTS = ["x_rr90", "x_ad", "x_gap", "x_lld365", "x_mag", "x_spat", "x_depth"]
TWO_SIDED = {"x_rr90", "x_lld365", "x_mag", "x_spat", "x_depth"}


def composite(F, panel, comps=COMPONENTS, name="x_surprise"):
    """Fisher-type total surprise: mean_k(-ln p_k), p_k from training-period empirical CDFs (pooled bins)."""
    key = panel[["bin_id", "t_start", "split"]]
    D = F.merge(key, on=["bin_id", "t_start"], how="left")
    trm = (D.split == "train").to_numpy()
    L = []
    for c in comps:
        x = D[c].to_numpy()
        ref = np.sort(x[trm & np.isfinite(x)])
        n = len(ref)
        lo = np.searchsorted(ref, x, side="right") / (n + 1)   # F(x)
        hi = 1 - np.searchsorted(ref, x, side="left") / (n + 1)  # P(X >= x)
        p = 2 * np.minimum(lo, hi) if c in TWO_SIDED else hi
        p = np.clip(p, 1 / (n + 1), 1.0)
        v = -np.log(p)
        v[~np.isfinite(x)] = np.nan
        L.append(v)
    L = np.vstack(L).T
    with np.errstate(all="ignore"):
        s = np.nanmean(L, axis=1)
    out = D[["bin_id", "t_start"]].copy()
    out[name] = s
    return out
