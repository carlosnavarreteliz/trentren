"""0.25-deg along-strike long-term Poisson count model: does static structure predict WHERE M>=6 occur in the test
period beyond smoothed past seismicity?

Run: /opt/miniconda3/bin/python3 -I count_model.py <profiles.csv> <out_json> [n_surr]

Honest forecasting scheme (no in-sample covariates):
  fit:      covariates (smoothed M>=4.7 rate) from window W1, response = target counts in window R1 (after W1)
  forecast: covariates from W2 = W1 U R1, response = counts in test window; mu scaled by duration ratio.
  main split: W1 = 1976-1994, R1 = 1995-2012, test = 2013-01-01..end.   alt split: W1 = 1976-1990, R1 = 1991-2004,
  test = 2005-2012.
Kernel bandwidth chosen on W1->R1 only.
Metrics: Poisson IG per test event (full - base) and spatial-only IG (normalized densities), both with
leave-block-out spatial CV variant; spatial FT surrogate of the structural profiles.
"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/structural_static")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.special import gammaln
import harness as H
import common as C

prof = C.load_profiles(sys.argv[1]); out_json = sys.argv[2]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
BD = 0.25
edges = np.arange(H.LAT_MIN, H.LAT_MAX + 1e-9, BD); nb = len(edges) - 1; centers = 0.5 * (edges[1:] + edges[:-1])
lat = prof.lat.to_numpy()
ev = H.get_events(min_mag=H.M_MODEL)
cat_end = H.load_catalog().time.max()
SIGMAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def yrs(a, b):
    return (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25


def smoothed_rate(t0, t1, sigma, mmin=H.M_MODEL):
    e = ev[(ev.time >= t0) & (ev.time < t1) & (ev.mag >= mmin)]
    la = e.latitude.to_numpy()
    w = norm.cdf((edges[1:][None, :] - la[:, None]) / sigma) - norm.cdf((edges[:-1][None, :] - la[:, None]) / sigma)
    r = w.sum(0) + 0.5 / nb  # weak floor
    return r / yrs(t0, t1)


def counts(t0, t1, mmin, main):
    e = ev[(ev.time >= t0) & (ev.time < t1) & (ev.mag >= mmin)]
    if main:
        e = e[e.mainshock]
    b = np.floor((e.latitude.to_numpy() - H.LAT_MIN) / BD).astype(int)
    b = b[(b >= 0) & (b < nb)]
    return np.bincount(b, minlength=nb).astype(float)


def pois_ll(y, mu):
    mu = np.maximum(mu, 1e-12)
    return np.sum(y * np.log(mu) - mu - gammaln(y + 1))


def fit_glm(X, y, l2=1e-2):
    """Poisson GLM log mu = X w (X includes intercept col), Newton with small ridge (not on intercept)."""
    w = np.zeros(X.shape[1]); w[0] = np.log(max(y.mean(), 1e-3))
    R = np.eye(X.shape[1]) * l2; R[0, 0] = 0
    for _ in range(100):
        mu = np.exp(np.clip(X @ w, -30, 30))
        g = X.T @ (y - mu) - R @ w
        Hs = X.T @ (X * mu[:, None]) + R
        step = np.linalg.solve(Hs, g)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def design(logr, F):
    return np.column_stack([np.ones(nb), logr] + ([F] if F is not None and F.shape[1] else []))


def run_split(W1, R1, TE, mmin, main, Fbins, sigma=None, cv_block=None):
    """Return dict with LLs for base / full and spatial-only IG. Fbins: [nb, k] standardized features (or None)."""
    y_fit = counts(R1[0], R1[1], mmin, main)
    y_te = counts(TE[0], TE[1], mmin, main)
    if sigma is None:  # choose bandwidth on fit split only
        lls = []
        for s in SIGMAS:
            r = smoothed_rate(W1[0], W1[1], s)
            c = y_fit.sum() / r.sum()
            lls.append(pois_ll(y_fit, c * r))
        sigma = SIGMAS[int(np.argmax(lls))]
    lr_fit = np.log(smoothed_rate(W1[0], W1[1], sigma))
    lr_te = np.log(smoothed_rate(W1[0], R1[1], sigma))
    scale = yrs(*TE) / yrs(*R1)
    out = {"sigma": sigma, "n_test": float(y_te.sum()), "n_fit": float(y_fit.sum())}
    for name, F in (("base", None), ("full", Fbins)):
        Xf, Xt = design(lr_fit, F), design(lr_te, F)
        if cv_block is None:
            w = fit_glm(Xf, y_fit)
            mu = np.exp(Xt @ w) * scale
        else:
            mu = np.empty(nb)
            blk = np.floor((centers - H.LAT_MIN) / cv_block).astype(int)
            for b in np.unique(blk):
                m = blk == b
                # exclude the block plus a 0.5-deg guard on each side from fitting
                guard = (centers >= centers[m].min() - 0.5) & (centers <= centers[m].max() + 0.5)
                w = fit_glm(Xf[~guard], y_fit[~guard])
                mu[m] = np.exp(Xt[m] @ w) * scale
        out[f"ll_{name}"] = pois_ll(y_te, mu)
        out[f"mu_{name}"] = mu
    N = max(y_te.sum(), 1)
    out["ig_pois"] = (out["ll_full"] - out["ll_base"]) / N
    pf, pb = out["mu_full"] / out["mu_full"].sum(), out["mu_base"] / out["mu_base"].sum()
    out["ig_spatial"] = float(np.sum(y_te * (np.log(pf) - np.log(pb))) / N)
    out["y_te"] = y_te
    return out


def feat_bins(vals):
    B = C.bin_profile(lat, vals, BD)
    B = np.where(np.isfinite(B), B, np.nanmean(B, 0))
    return (B - B.mean(0)) / np.where(B.std(0) > 0, B.std(0), 1)


SPLITS = {"main": (("1976-01-01", "1995-01-01"), ("1995-01-01", "2013-01-01"), ("2013-01-01", cat_end)),
          "alt2005": (("1976-01-01", "1991-01-01"), ("1991-01-01", "2005-01-01"), ("2005-01-01", "2013-01-01"))}
vals = prof[C.PRIMARY].to_numpy(float)
F0 = feat_bins(vals)
results = {}
rng = np.random.default_rng(11)
for split, (W1, R1, TE) in SPLITS.items():
    for mmin in (6.0, 5.5):
        for main in (True, False):
            for cv in (None, 3.0):
                key = f"{split}_M{mmin}_{'main' if main else 'all'}_{'cv3' if cv else 'nocv'}"
                r = run_split(W1, R1, TE, mmin, main, F0, cv_block=cv)
                nsur = NS if (split == "main" and mmin == 6.0) else max(NS // 5, 100)
                nul_p, nul_s = np.empty(nsur), np.empty(nsur)
                for i in range(nsur):
                    rs = run_split(W1, R1, TE, mmin, main, feat_bins(C.spatial_ft(vals, rng)), sigma=r["sigma"],
                                   cv_block=cv)
                    nul_p[i], nul_s[i] = rs["ig_pois"], rs["ig_spatial"]
                results[key] = {"sigma": r["sigma"], "n_test": r["n_test"], "n_fit": r["n_fit"],
                                "ig_pois": float(r["ig_pois"]), "ig_spatial": r["ig_spatial"],
                                "ll_base": float(r["ll_base"]), "ll_full": float(r["ll_full"]),
                                "null_pois": C.pval(r["ig_pois"], nul_p), "null_spatial": C.pval(r["ig_spatial"], nul_s)}
                print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in results[key].items()
                            if not k.startswith("null")},
                      "p_pois=%.4f p_spat=%.4f" % (results[key]["null_pois"]["p_value"],
                                                   results[key]["null_spatial"]["p_value"]), flush=True)
                if split == "main" and mmin == 6.0 and main and cv is None:
                    keep = r

# --- M>=7 descriptive: where do 2013+ M>=7 epicentres fall in the rank order of each model's rate? -------
e7 = ev[(ev.time >= "2013-01-01") & (ev.mag >= 7.0)]
b7 = np.floor((e7.latitude.to_numpy() - H.LAT_MIN) / BD).astype(int)
rank = lambda mu, b: float((mu < mu[b]).mean())  # fraction of bins with lower rate
results["m7_test_events"] = [{"time": str(t), "mag": float(m), "lat": float(la), "bin": int(b),
                              "pct_rank_base": rank(keep["mu_base"], b), "pct_rank_full": rank(keep["mu_full"], b)}
                             for t, m, la, b in zip(e7.time, e7.mag, e7.latitude, b7) if 0 <= b < nb]
for d in results["m7_test_events"]:
    print("M7", d)
# coefficient report for the main model
lr_fit = np.log(smoothed_rate("1976-01-01", "1995-01-01", keep["sigma"]))
w = fit_glm(design(lr_fit, F0), counts("1995-01-01", "2013-01-01", 6.0, True))
results["coef_main_M6_main"] = dict(zip(["intercept", "log_bg_rate"] + C.PRIMARY, np.round(w, 4).tolist()))
print("coef", results["coef_main_M6_main"])
results["profile_bins_0p25"] = {"centers": centers.tolist(), "mu_base": keep["mu_base"].tolist(),
                                "mu_full": keep["mu_full"].tolist(), "y_test": keep["y_te"].tolist()}
json.dump(results, open(out_json, "w"), indent=1, default=float)
print("saved", out_json)
