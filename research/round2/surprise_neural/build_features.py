"""Build causal per-bin weekly features from the cross-fitted NTPP and the cross-fitted autoencoder.
Usage: python3 -I build_features.py MODELDIR OUTPARQUET"""
import sys, os, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/surprise_neural")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import harness as H
import ntpp_lib as L

torch.set_num_threads(2)
MD, OUTP = sys.argv[1], sys.argv[2]
BLOCKS = {"f0": (1973, 1982), "f1": (1983, 1992), "f2": (1993, 2002), "f3": (2003, 2012), "final": (2013, 2100)}
SEEDS = (0, 1, 2)
NB = 29

ev = H.get_events(min_mag=4.7)
D = L.prep_events(ev)
t, n = D["t"], D["n"]
ebin = D["bin_id"]
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
issues = np.sort(panel.t_start.unique())
T0 = H.to_days(pd.DatetimeIndex(issues))
iyear = pd.DatetimeIndex(issues).year.to_numpy()
NI = len(issues)


def nbr3(A):
    """sum over bins b-1..b+1 along last axis."""
    P = np.pad(A, [(0, 0)] * (A.ndim - 1) + [(1, 1)])
    return P[..., :-2] + P[..., 1:-1] + P[..., 2:]


# ------------------------------------------------------------------ NTPP features
F = {k: np.full((NI, NB), np.nan) for k in
     ["x_ntpp_logp60", "x_ntpp_logp55", "x_resid_30", "x_resid_90", "x_resid_365", "x_lam_90",
      "x_surp_mean_365", "x_surp_mag_365", "x_surp_time_365", "x_upd_norm_365"]}
fc = np.load(os.path.join(MD, "forecast_weekly.npz"))
assert np.all(fc["issues"].astype("datetime64[ns]") == issues.astype("datetime64[ns]"))
F["x_ntpp_logp55"] = np.log10(np.clip(fc["P"][:, 0, :], 1e-6, 1))
F["x_ntpp_logp60"] = np.log10(np.clip(fc["P"][:, 1, :], 1e-6, 1))

# observed counts in bins b-1..b+1 over windows (strictly before t0)
order_by_bin = [np.sort(t[ebin == b]) for b in range(NB)]


def obs_count(A, B):
    c = np.zeros((len(A), NB))
    for b in range(NB):
        tb = order_by_bin[b]
        c[:, b] = np.searchsorted(tb, B, "left") - np.searchsorted(tb, A, "left")
    return nbr3(c)


tau_next = np.r_[D["tau"][1:], np.nan]
for k, (lo, hi) in BLOCKS.items():
    sel = np.where((iyear >= lo) & (iyear <= hi))[0]
    if len(sel) == 0:
        continue
    Tk = T0[sel]
    lam = {W: np.zeros((len(sel), NB)) for W in (30, 90, 365)}
    nll_tot = np.zeros(n); nll_m = np.zeros(n); nll_t = np.zeros(n); upd = np.zeros(n)
    for s in SEEDS:
        sc = np.load(os.path.join(MD, f"score_{k}_s{s}.npz"))
        tp = {"tw": sc["tw"], "tmu": sc["tmu"], "tsig": sc["tsig"]}
        P3 = nbr3(sc["latp"])                                     # [n, 29]
        Cfull = -L.log_surv_np(tp, np.arange(n - 1), tau_next[:-1])   # [n-1]
        cum = np.zeros((n + 1, NB))
        cum[1:n] = np.cumsum(P3[:-1] * Cfull[:, None], axis=0)

        def Lam(T):
            iT = np.searchsorted(t, T, "left") - 1
            out = np.zeros((len(T), NB))
            ok = iT >= 0
            ii = iT[ok]
            part = -L.log_surv_np(tp, ii, T[ok] - t[ii])
            out[ok] = cum[ii] + P3[ii] * part[:, None]
            return out
        LT = Lam(Tk)
        for W in lam:
            lam[W] += (LT - Lam(Tk - W)) / len(SEEDS)
        nll_tot += (sc["nll_t"] + sc["nll_m"] + sc["nll_l"]) / len(SEEDS)
        nll_m += sc["nll_m"] / len(SEEDS)
        nll_t += sc["nll_t"] / len(SEEDS)
        upd += sc["upd"] / len(SEEDS)
    for W in (30, 90, 365):
        N = obs_count(Tk - W, Tk)
        F[f"x_resid_{W}"][sel] = np.log((N + 0.5) / (lam[W] + 0.5))
        if W == 90:
            F["x_lam_90"][sel] = np.log10(lam[W] + 1e-3)
    # surprise means over events in bins b-1..b+1, [t0-365, t0)
    for name, arr in [("x_surp_mean_365", nll_tot), ("x_surp_mag_365", nll_m), ("x_surp_time_365", nll_t),
                      ("x_upd_norm_365", upd)]:
        S = np.zeros((len(sel), NB)); C = np.zeros((len(sel), NB))
        a = np.nan_to_num(arr)
        for b in range(NB):
            idx = np.where(ebin == b)[0]
            cs = np.r_[0, np.cumsum(a[idx])]
            tb = t[idx]
            i1 = np.searchsorted(tb, Tk, "left"); i0 = np.searchsorted(tb, Tk - 365, "left")
            S[:, b] = cs[i1] - cs[i0]; C[:, b] = i1 - i0
        S3, C3 = nbr3(S), nbr3(C)
        F[name][sel] = np.where(C3 > 0, S3 / np.maximum(C3, 1), np.nan)
    print("ntpp features", k, len(sel), flush=True)

# ------------------------------------------------------------------ autoencoder of activity vectors
# weekly counts per bin in [t0-7, t0)
Wk = np.zeros((NI, NB))
for b in range(NB):
    tb = order_by_bin[b]
    Wk[:, b] = np.searchsorted(tb, T0, "left") - np.searchsorted(tb, T0 - 7, "left")
Y52 = np.zeros((NI, NB))
for b in range(NB):
    tb = order_by_bin[b]
    Y52[:, b] = np.searchsorted(tb, T0, "left") - np.searchsorted(tb, T0 - 364, "left")
lagW = np.stack([np.vstack([np.zeros((j, NB)), Wk[:NI - j]]) for j in range(8)], -1)   # [NI, NB, 8] lag j weeks
padW = np.pad(lagW, [(0, 0), (2, 2), (0, 0)])
padY = np.pad(Y52, [(0, 0), (2, 2)])
V = np.zeros((NI, NB, 45), np.float32)
for b in range(NB):
    V[:, b, :40] = np.log1p(padW[:, b:b + 5, :]).reshape(NI, 40)
    V[:, b, 40:] = np.log1p(padY[:, b:b + 5])
row_year = np.repeat(iyear[:, None], NB, 1)
valid_hist = (T0 - H.to_days(pd.Timestamp("1973-01-01")) >= 364)[:, None] & np.ones((1, NB), bool)


class AE(nn.Module):
    def __init__(self, d=45):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 4))
        self.d = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, d))

    def forward(self, x):
        return self.d(self.e(x))


def fit_ae(Xtr, epochs, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    X = torch.tensor((Xtr - mu) / sd)
    m = AE()
    opt = torch.optim.Adam(m.parameters(), 2e-3, weight_decay=1e-5)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            xb = X[perm[i:i + 512]]
            loss = ((m(xb + 0.1 * torch.randn_like(xb)) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m, mu, sd


@torch.no_grad()
def score_ae(m, mu, sd, X):
    Z = torch.tensor((X - mu) / sd)
    return np.log(((m(Z) - Z) ** 2).mean(1).numpy() + 1e-6)


flatV = V.reshape(-1, 45)
fy = row_year.reshape(-1)
fv = valid_hist.reshape(-1)
# epoch selection: fit < 2008, validate 2008-2012 (mean val reconstruction MSE)
sel_log = {}
trm = fv & (fy <= 2007); vam = fv & (fy >= 2008) & (fy <= 2012)
for ep in (5, 15, 40):
    m, mu, sd = fit_ae(flatV[trm], ep)
    sel_log[ep] = float(np.mean(np.exp(score_ae(m, mu, sd, flatV[vam]))))
    print("AE epochs", ep, "val mse", sel_log[ep], flush=True)
EPA = min(sel_log, key=sel_log.get)
ae_score = np.full(len(flatV), np.nan)
for k, (lo, hi) in BLOCKS.items():
    held = (fy >= lo) & (fy <= hi)
    trk = fv & (fy <= 2012) & ~held
    sc = np.zeros(held.sum())
    for s in SEEDS:
        m, mu, sd = fit_ae(flatV[trk], EPA, seed=s)
        sc += score_ae(m, mu, sd, flatV[held]) / len(SEEDS)
    ae_score[held] = sc
    print("AE fold", k, flush=True)
ae_score[~fv] = np.nan
F["x_ae_score"] = ae_score.reshape(NI, NB)

# ------------------------------------------------------------------ assemble
TT, BB = np.meshgrid(np.arange(NI), np.arange(NB), indexing="ij")
df = pd.DataFrame({"bin_id": BB.ravel().astype(np.int16), "t_start": issues[TT.ravel()]})
for k, v in F.items():
    df[k] = v.ravel()
df.to_parquet(OUTP, index=False)
json.dump({"ae_epoch_selection": sel_log, "ae_epochs": EPA}, open(os.path.join(MD, "ae_selection.json"), "w"))
print(df.describe().T.to_string())
