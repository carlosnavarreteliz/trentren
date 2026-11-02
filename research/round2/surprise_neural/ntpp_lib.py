"""Small RECAST-like neural temporal point process for the Chilean M>=4.7 catalog.

Intensity-free formulation (Shchur et al. 2020): a GRU summarizes the event history; given the state after
event i, the next inter-event time is a log-normal mixture, the magnitude excess a mixture of exponentials and the
(normalized) latitude a truncated Gaussian mixture.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(2)

LAT_LO, LAT_HI = -46.0, -17.0
LAT_C, LAT_S = -31.5, 14.5          # z = (lat - LAT_C)/LAT_S in [-1, 1]
M0 = 4.65                           # continuous magnitude origin (catalog M>=4.7 rounded to 0.1)
TAU_FLOOR = 1e-4                    # days
KT, KM, KL = 8, 3, 8
EDGES_Z = torch.tensor((np.arange(LAT_LO, LAT_HI + 1e-9, 1.0) - LAT_C) / LAT_S, dtype=torch.float32)
SQ2 = math.sqrt(2.0)


def ncdf(x):
    return 0.5 * (1 + torch.erf(x / SQ2))


def log_ncdf_upper(z):
    """log(1 - Phi(z)) stable."""
    return torch.special.log_ndtr(-z)


def prep_events(ev: pd.DataFrame) -> dict:
    ev = ev.sort_values("time").reset_index(drop=True)
    t = ev.t_days.to_numpy(float)
    tau = np.diff(t, prepend=t[0] - 1.0)
    tau = np.maximum(tau, TAU_FLOOR)
    m = ev.mag.to_numpy(float)
    lat = ev.latitude.to_numpy(float)
    lon = ev.longitude.to_numpy(float)
    dep = ev.depth.to_numpy(float)
    X = np.column_stack([np.log10(tau) / 1.5, m - M0, (lat - LAT_C) / LAT_S, (lon - (-72.0)) / 3.0,
                         (np.log1p(np.maximum(dep, 0)) - 3.8) / 0.8]).astype(np.float32)
    return dict(t=t, tau=tau, m=m, lat=lat, lon=lon, dep=dep, X=X, time=ev.time.to_numpy(), n=len(ev),
                bin_id=ev.bin_id.to_numpy(int), year=ev.time.dt.year.to_numpy(int))


class NTPP(nn.Module):
    def __init__(self, H=64, din=5):
        super().__init__()
        self.H = H
        self.gru = nn.GRU(din, H, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(H, 64), nn.Tanh())
        self.t_head = nn.Linear(64, 3 * KT)
        self.m_head = nn.Linear(64, 2 * KM)
        self.l_head = nn.Linear(64, 3 * KL)
        self.h0 = nn.Parameter(torch.zeros(1, 1, H))
        self.eps_head = nn.Linear(64, 2)                       # defensive-mixture weights (mag, lat)
        self.register_buffer("hist_mass", torch.full((29,), 1 / 29.))   # train latitude histogram (bin masses)
        self.register_buffer("beta_gr", torch.tensor(2.0))               # train GR rate (1/mean excess)

    def set_reference(self, lat, m):
        cnt = np.bincount(np.clip(np.floor(lat - LAT_LO).astype(int), 0, 28), minlength=29).astype(float) + 0.5
        self.hist_mass.copy_(torch.tensor(cnt / cnt.sum(), dtype=torch.float32))
        self.beta_gr.copy_(torch.tensor(1.0 / np.mean(m - M0), dtype=torch.float32))

    def params(self, h):
        z = self.mlp(h)
        a = self.t_head(z)
        tw = torch.log_softmax(a[..., :KT], -1)
        tmu = a[..., KT:2 * KT] + torch.linspace(-6, 3, KT)       # spread init over log-days
        tsig = nn.functional.softplus(a[..., 2 * KT:] + 0.5) + 0.05
        b = self.m_head(z)
        mw = torch.log_softmax(b[..., :KM], -1)
        mrate = nn.functional.softplus(b[..., KM:] + torch.tensor([2.0, 1.0, 0.5])) + 0.2
        c = self.l_head(z)
        lw = torch.log_softmax(c[..., :KL], -1)
        lmu = torch.tanh(c[..., KL:2 * KL] + torch.linspace(-1, 1, KL)) * 1.1
        lsig = nn.functional.softplus(c[..., 2 * KL:] - 1.5) + 0.02
        e = torch.sigmoid(self.eps_head(z) - 2.0) * 0.5 + 0.02
        return dict(tw=tw, tmu=tmu, tsig=tsig, mw=mw, mrate=mrate, lw=lw, lmu=lmu, lsig=lsig,
                    eps_m=e[..., 0], eps_l=e[..., 1])

    # ---- per-component log likelihoods --------------------------------------------------------------
    @staticmethod
    def nll_time(p, tau):
        y = torch.log(tau).unsqueeze(-1)
        lp = p["tw"] - 0.5 * ((y - p["tmu"]) / p["tsig"]) ** 2 - torch.log(p["tsig"]) - 0.5 * math.log(2 * math.pi)
        return -(torch.logsumexp(lp, -1) - y.squeeze(-1))

    @staticmethod
    def log_surv(p, dt):
        """log S(dt) for dt>=0 (dt tensor broadcastable to batch)."""
        y = torch.log(torch.clamp(dt, min=1e-12)).unsqueeze(-1)
        ls = p["tw"] + log_ncdf_upper((y - p["tmu"]) / p["tsig"])
        out = torch.logsumexp(ls, -1)
        return torch.where(dt <= 0, torch.zeros_like(out), out)

    def nll_mag(self, p, mex):
        x = torch.clamp(mex, min=0).unsqueeze(-1)
        lp = p["mw"] + torch.log(p["mrate"]) - p["mrate"] * x
        ln = torch.logsumexp(lp, -1)
        lg = torch.log(self.beta_gr) - self.beta_gr * x.squeeze(-1)
        return -torch.logaddexp(torch.log1p(-p["eps_m"]) + ln, torch.log(p["eps_m"]) + lg)

    def log_p_mag_ge(self, p, thr):
        ln = torch.logsumexp(p["mw"] - p["mrate"] * thr, -1)
        lg = -self.beta_gr * thr
        return torch.logaddexp(torch.log1p(-p["eps_m"]) + ln, torch.log(p["eps_m"]) + lg)

    @staticmethod
    def _lat_norm(p):
        a = (1.0 - p["lmu"]) / p["lsig"]
        b = (-1.0 - p["lmu"]) / p["lsig"]
        return torch.clamp(ncdf(a) - ncdf(b), min=1e-8)

    def nll_lat(self, p, z):
        zz = z.unsqueeze(-1)
        Z = self._lat_norm(p)
        lp = p["lw"] - 0.5 * ((zz - p["lmu"]) / p["lsig"]) ** 2 - torch.log(p["lsig"]) - 0.5 * math.log(2 * math.pi) \
            - torch.log(Z)
        ln = torch.logsumexp(lp, -1)
        bi = torch.clamp(((z + 1) / 2 * 29).long(), 0, 28)
        lh = torch.log(self.hist_mass[bi] * 29 / 2)              # density in z units
        return -torch.logaddexp(torch.log1p(-p["eps_l"]) + ln, torch.log(p["eps_l"]) + lh)

    def lat_bin_probs(self, p):
        """P(lat in each 1-degree bin) -> [..., 29]."""
        e = EDGES_Z.view(*([1] * p["lmu"].dim()), -1)          # [...,1,30]
        mu, sg = p["lmu"].unsqueeze(-1), p["lsig"].unsqueeze(-1)
        cdf = ncdf((e - mu) / sg)                              # [..., KL, 30]
        mass = cdf[..., 1:] - cdf[..., :-1]                    # [..., KL, 29]
        mass = mass / self._lat_norm(p).unsqueeze(-1)
        pn = (p["lw"].exp().unsqueeze(-1) * mass).sum(-2)
        el = p["eps_l"].unsqueeze(-1)
        return (1 - el) * pn + el * self.hist_mass

    def nll_terms(self, p, tau, mex, z):
        return self.nll_time(p, tau), self.nll_mag(p, mex), self.nll_lat(p, z)


# ---------------------------------------------------------------------------------------------------------
def make_streams(idx_segments, n_streams=8):
    """Split each training segment (array of event indices i, each predicting i+1) into contiguous streams."""
    streams = []
    tot = sum(len(s) for s in idx_segments)
    target_len = max(64, tot // n_streams)
    for s in idx_segments:
        k = max(1, int(round(len(s) / target_len)))
        for part in np.array_split(s, k):
            if len(part) > 8:
                streams.append(part)
    L = max(len(s) for s in streams)
    I = np.zeros((len(streams), L), int)
    M = np.zeros((len(streams), L), bool)
    for r, s in enumerate(streams):
        I[r, :len(s)] = s
        M[r, :len(s)] = True
    return I, M


def train_model(D, train_mask_event, H=64, epochs=40, lr=3e-3, wd=1e-4, chunk=128, seed=0, val_fn=None,
                val_every=2, verbose=False, ema=0.98):
    """train_mask_event[i] True if event i AND its successor belong to the same training segment (i predicts i+1)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = D["n"]
    ok = np.zeros(n, bool)
    ok[:-1] = train_mask_event[:-1] & train_mask_event[1:]
    # segments = runs of consecutive ok indices
    idx = np.where(ok)[0]
    segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    I, Mk = make_streams(segs)
    X = torch.tensor(D["X"])
    tau_next = torch.tensor(np.r_[D["tau"][1:], 1.0], dtype=torch.float32)
    mex_next = torch.tensor(np.r_[D["m"][1:] - M0, 0.1], dtype=torch.float32)
    z_next = torch.tensor(np.r_[(D["lat"][1:] - LAT_C) / LAT_S, 0.0], dtype=torch.float32)
    model = NTPP(H)
    model.set_reference(D["lat"][train_mask_event], D["m"][train_mask_event])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n_chunks = int(np.ceil(I.shape[1] / chunk))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * n_chunks, eta_min=lr * 0.05)
    import copy
    ema_model = copy.deepcopy(model)
    It = torch.tensor(I)
    Mt = torch.tensor(Mk)
    hist = []
    best = (np.inf, None, 0)
    for ep in range(1, epochs + 1):
        model.train()
        h = model.h0.expand(1, I.shape[0], H).contiguous()
        tot, cnt = 0.0, 0
        for s0 in range(0, I.shape[1], chunk):
            ii = It[:, s0:s0 + chunk]
            mm = Mt[:, s0:s0 + chunk]
            out, h = model.gru(X[ii], h.detach())
            p = model.params(out)
            a, b, c = model.nll_terms(p, tau_next[ii], mex_next[ii], z_next[ii])
            nll = a + b + c
            loss = (nll * mm).sum() / mm.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema_model.parameters(), model.parameters()):
                    pe.mul_(ema).add_(pm.detach(), alpha=1 - ema)
            tot += float((nll * mm).sum().detach())
            cnt += int(mm.sum())
        rec = {"epoch": ep, "train_nll": tot / cnt}
        if val_fn is not None and ep % val_every == 0:
            v = val_fn(ema_model)
            rec["val_nll"] = v
            if v < best[0]:
                best = (v, {k: t.clone() for k, t in ema_model.state_dict().items()}, ep)
        hist.append(rec)
        if verbose:
            print(rec, flush=True)
    return ema_model, hist, best


@torch.no_grad()
def score_sequence(model, D):
    """Run the GRU causally over the full sequence. Returns per-event arrays (index i = state AFTER event i,
    predicting event i+1) and per-event NLL of event j given state j-1."""
    model.eval()
    X = torch.tensor(D["X"]).unsqueeze(0)
    out, _ = model.gru(X, model.h0)
    Hs = out[0]                                         # [n, H] state after event i
    p = model.params(Hs)
    n = D["n"]
    tau_next = torch.tensor(np.r_[D["tau"][1:], 1.0], dtype=torch.float32)
    mex_next = torch.tensor(np.r_[D["m"][1:] - M0, 0.1], dtype=torch.float32)
    z_next = torch.tensor(np.r_[(D["lat"][1:] - LAT_C) / LAT_S, 0.0], dtype=torch.float32)
    a, b, c = model.nll_terms(p, tau_next, mex_next, z_next)
    # NLL of event j given state j-1 -> shift by one (event 0 has no prediction)
    nll_t = np.r_[np.nan, a.numpy()[:-1]]
    nll_m = np.r_[np.nan, b.numpy()[:-1]]
    nll_l = np.r_[np.nan, c.numpy()[:-1]]
    h_prev = torch.cat([model.h0[0], Hs[:-1]], 0)
    upd = (Hs - h_prev).norm(dim=-1).numpy()
    latp = model.lat_bin_probs(p).numpy()               # [n, 29] for next event given state i
    pm55 = model.log_p_mag_ge(p, torch.tensor(5.45 - M0)).exp().numpy()
    pm60 = model.log_p_mag_ge(p, torch.tensor(5.95 - M0)).exp().numpy()
    tparams = {k: p[k].numpy() for k in ("tw", "tmu", "tsig")}
    return dict(nll_t=nll_t, nll_m=nll_m, nll_l=nll_l, upd=upd, latp=latp, pm55=pm55, pm60=pm60,
                tparams=tparams, H=Hs.numpy())


def log_surv_np(tp, i, dt):
    """log S_i(dt) for arrays i (event index) and dt (days)."""
    from scipy.special import log_ndtr, logsumexp
    dt = np.asarray(dt, float)
    y = np.log(np.maximum(dt, 1e-12))[:, None]
    ls = tp["tw"][i] + log_ndtr(-(y - tp["tmu"][i]) / tp["tsig"][i])
    out = logsumexp(ls, axis=1)
    return np.where(dt <= 0, 0.0, out)


# ---------------------------------------------------------------------------------------------------------
class LonDepSampler:
    """Draw (lon, depth) of a simulated event from training events with a similar latitude."""

    def __init__(self, D, mask, width=0.5):
        lat, lon, dep = D["lat"][mask], D["lon"][mask], D["dep"][mask]
        self.width = width
        nb = int(round((LAT_HI - LAT_LO) / width))
        fb = np.clip(((lat - LAT_LO) / width).astype(int), 0, nb - 1)
        order = np.argsort(fb, kind="stable")
        self.lon = torch.tensor(lon[order], dtype=torch.float32)
        self.dep = torch.tensor(dep[order], dtype=torch.float32)
        cnt = np.bincount(fb, minlength=nb)
        start = np.r_[0, np.cumsum(cnt)[:-1]]
        # empty fine bins -> nearest non-empty
        ne = np.where(cnt > 0)[0]
        near = ne[np.abs(np.arange(nb)[:, None] - ne[None, :]).argmin(1)]
        self.start = torch.tensor(start[near])
        self.cnt = torch.tensor(cnt[near])
        self.nb = nb

    def sample(self, lat):
        fb = torch.clamp(((lat - LAT_LO) / self.width).long(), 0, self.nb - 1)
        u = torch.rand(lat.shape)
        j = self.start[fb] + (u * self.cnt[fb]).long() % self.cnt[fb]
        return self.lon[j], self.dep[j]


@torch.no_grad()
def simulate_forecasts(model, D, Hs, issue_days, horizon, thresholds, sampler, n_paths=300, max_steps=300,
                       batch_issues=16, seed=0):
    """Rao-Blackwellized P(>=1 event with M>=thr in each 1-deg bin within [t0, t0+horizon)).
    Hs: [n, H] states after each event (from score_sequence). Returns array [len(issue), len(thr), 29] and
    number of truncated paths."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    t = D["t"]
    Ht = torch.tensor(Hs)
    thr_ex = torch.tensor([th - 0.05 - M0 for th in thresholds], dtype=torch.float32)
    out = np.zeros((len(issue_days), len(thresholds), 29))
    n_trunc = 0
    for s in range(0, len(issue_days), batch_issues):
        T0 = np.asarray(issue_days[s:s + batch_issues], float)
        nI = len(T0)
        last = np.searchsorted(t, T0, side="left") - 1           # last event strictly before t0
        B = nI * n_paths
        h = Ht[np.repeat(last, n_paths)].clone()                  # [B, H]
        tlast = torch.tensor(np.repeat(t[last] - T0, n_paths), dtype=torch.float32)  # relative to t0 (<0)
        lowb = -tlast.clone()                                     # first step: tau > elapsed
        logS = torch.zeros(B, len(thresholds), 29)
        active = torch.ones(B, dtype=torch.bool)
        for step in range(max_steps):
            ia = torch.nonzero(active).squeeze(1)
            if len(ia) == 0:
                break
            hh = h[ia]
            p = model.params(hh)
            L = lowb[ia]
            # component posterior given tau > L
            yL = torch.log(torch.clamp(L, min=1e-12)).unsqueeze(-1)
            zL = (yL - p["tmu"]) / p["tsig"]
            lsurv = torch.where(L.unsqueeze(-1) > 0, log_ncdf_upper(zL), torch.zeros_like(zL))
            post = p["tw"] + lsurv
            k = torch.multinomial(torch.softmax(post, -1), 1).squeeze(1)
            mu = p["tmu"].gather(1, k[:, None]).squeeze(1)
            sg = p["tsig"].gather(1, k[:, None]).squeeze(1)
            zl = torch.where(L > 0, (torch.log(torch.clamp(L, min=1e-12)) - mu) / sg, torch.full_like(mu, -40.0))
            Fl = ncdf(zl)
            u = Fl + torch.rand(len(ia)) * (1 - Fl)
            u = torch.clamp(u, 1e-7, 1 - 1e-7)
            y = mu + sg * torch.special.ndtri(u)
            y = torch.maximum(y, torch.log(torch.clamp(L, min=1e-12)) + 1e-6)
            tau = torch.clamp(torch.exp(y), min=TAU_FLOOR)
            tnew = tlast[ia] + tau
            inwin = (tnew >= 0) & (tnew < horizon)
            ended = tnew >= horizon
            # Rao-Blackwell contribution for events inside the window
            if inwin.any():
                pw = {kk: v[inwin] for kk, v in p.items()}
                pm = torch.stack([model.log_p_mag_ge(pw, th).exp() for th in thr_ex], 1)  # [b, nthr]
                lb = model.lat_bin_probs(pw)                                                 # [b, 29]
                q = torch.clamp(pm.unsqueeze(-1) * lb.unsqueeze(1), max=1 - 1e-7)
                logS[ia[inwin]] += torch.log1p(-q)
            # continue only events inside the window (events before t0 cannot occur: tau > elapsed)
            cont = inwin
            ic = ia[cont]
            if len(ic):
                pc = {kk: v[cont] for kk, v in p.items()}
                # sample magnitude
                km = torch.multinomial(pc["mw"].exp(), 1).squeeze(1)
                rate = pc["mrate"].gather(1, km[:, None]).squeeze(1)
                usegr = torch.rand(len(ic)) < pc["eps_m"]
                rate = torch.where(usegr, model.beta_gr.expand_as(rate), rate)
                mex = -torch.log(torch.clamp(torch.rand(len(ic)), min=1e-12)) / rate
                # sample latitude (truncated mixture) by rejection
                kl = torch.multinomial(pc["lw"].exp(), 1).squeeze(1)
                lm = pc["lmu"].gather(1, kl[:, None]).squeeze(1)
                ls = pc["lsig"].gather(1, kl[:, None]).squeeze(1)
                Fa, Fb = ncdf((-1 - lm) / ls), ncdf((1 - lm) / ls)
                uu = torch.clamp(Fa + torch.rand(len(ic)) * (Fb - Fa), 1e-7, 1 - 1e-7)
                zz = torch.clamp(lm + ls * torch.special.ndtri(uu), -1, 1)
                useh = torch.rand(len(ic)) < pc["eps_l"]
                if useh.any():
                    hb = torch.multinomial(model.hist_mass, int(useh.sum()), replacement=True)
                    latb = LAT_LO + hb.float() + torch.rand(len(hb))
                    zz[useh] = (latb - LAT_C) / LAT_S
                lat = zz * LAT_S + LAT_C
                lon, dep = sampler.sample(lat)
                taus = tau[cont]
                x = torch.stack([torch.log10(taus) / 1.5, mex + 0.0, zz, (lon + 72.0) / 3.0,
                                 (torch.log1p(torch.clamp(dep, min=0)) - 3.8) / 0.8], 1)
                _, hn = model.gru(x.unsqueeze(1), hh[cont].unsqueeze(0).contiguous())
                h[ic] = hn[0]
                tlast[ic] = tnew[cont]
                lowb[ic] = 0.0
            active[ia[ended]] = False
        n_trunc += int(active.sum())
        P = 1 - torch.exp(logS).view(nI, n_paths, len(thresholds), 29).mean(1)
        out[s:s + nI] = P.numpy()
    return out, n_trunc
