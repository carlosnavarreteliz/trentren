"""Marchitelli et al. (2020) in its ORIGINAL (non-forecast) form + superposed-epoch analysis for Chile.

usage: python -I run_epoch.py <space_weather_daily.csv> <global_m56.csv> <out_json> [n_null=5000]

(a) GLOBAL: daily count of USGS M>=5.6 earthquakes on day D vs log10 Np on day D-1 (Spearman rho, and rate ratio
    of days following top-decile Np days vs the rest). Periods: 1996-2016 (their study window), 2017-2026
    (out-of-sample), 1996-2026. Null: circular shift of the Np series by >= 365 days (earthquake series untouched,
    so its clustering/autocorrelation is preserved).
(b) CHILE: superposed epoch: mean log10 Np on days D-k (k = 1..7) before Chilean M>=6 events (all / ZBZ mainshocks),
    1976-2026 and 2013-2026, vs the same circular-shift null.
"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/space_weather")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
import harness as H
import sw_lib as L

csv, gcsv, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
NN = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
rng = np.random.default_rng(7)
daily = L.load_daily(csv)
x = daily["lognp"]
days = daily.index
res = {}

# ---------------- (a) global replication ---------------------------------------------------------------
g = pd.read_csv(gcsv)
g = g[g["type"] == "earthquake"]
gt = pd.to_datetime(g["time"], utc=True).dt.tz_convert(None).dt.floor("D")
cnt = gt.value_counts().reindex(days, fill_value=0).astype(float)
xl = x.shift(1)                          # Np on day D-1 aligned to day D


def stat(xv, cv):
    ok = np.isfinite(xv)
    rho = stats.spearmanr(xv[ok], cv[ok]).statistic
    thr = np.nanquantile(xv[ok], 0.9)
    hi = xv[ok] >= thr
    rr = cv[ok][hi].mean() / cv[ok][~hi].mean()
    return rho, rr


for lab, (a, b) in {"1996-2016": ("1996-01-01", "2016-12-31"), "2017-2026": ("2017-01-01", "2026-10-17"),
                    "1996-2026": ("1996-01-01", "2026-10-17")}.items():
    sl = slice(a, b)
    xv, cv = xl.loc[sl].to_numpy(), cnt.loc[sl].to_numpy()
    rho, rr = stat(xv, cv)
    T = len(xv)
    xr = x.loc[sl].to_numpy()
    nr, nrr = [], []
    for _ in range(NN // 5):
        k = rng.integers(365, T - 365)
        xs = np.roll(xr, k)
        xs = np.concatenate([[np.nan], xs[:-1]])       # same D-1 alignment
        r_, rr_ = stat(xs, cv)
        nr.append(r_); nrr.append(rr_)
    nr, nrr = np.array(nr), np.array(nrr)
    res[f"global_{lab}"] = {"n_days": int(T), "n_eq": int(cv.sum()), "spearman_rho": float(rho),
                            "p_rho_shift": float((1 + np.sum(nr >= rho)) / (len(nr) + 1)),
                            "null_rho_sd": float(nr.std()),
                            "rate_ratio_top10pct": float(rr),
                            "p_rr_shift": float((1 + np.sum(nrr >= rr)) / (len(nrr) + 1))}
    print("GLOBAL", lab, res[f"global_{lab}"], flush=True)

# lag scan, global 1996-2026 (descriptive)
lagscan = {}
for lag in range(-5, 6):
    xv = x.shift(lag).loc["1996":"2026"].to_numpy()
    ok = np.isfinite(xv)
    lagscan[lag] = float(stats.spearmanr(xv[ok], cnt.loc["1996":"2026"].to_numpy()[ok]).statistic)
res["global_lagscan_rho"] = lagscan
print("lag scan (positive lag = Np leads):", {k: round(v, 4) for k, v in lagscan.items()})

# ---------------- (b) Chile superposed epoch --------------------------------------------------------------
ev = H.get_events(min_mag=6.0, start="1976-01-01")
xa = x.to_numpy()
day_idx = pd.Series(np.arange(len(days)), index=days)
for sel_lab, sel in {"all": np.ones(len(ev), bool), "main": ev["mainshock"].to_numpy(bool)}.items():
    for per, a in {"1976-2026": "1976-01-01", "2013-2026": "2013-01-01"}.items():
        e = ev[sel & (ev.time >= a)]
        di = day_idx.reindex(e.time.dt.floor("D")).to_numpy()
        di = di[np.isfinite(di)].astype(int)
        out = {"n_events": int(len(di))}
        for k in (1, 2, 3, 7):
            idx = di - k
            obs = np.nanmean(xa[idx])
            null = []
            for _ in range(NN):
                s = rng.integers(365, len(xa) - 365)
                null.append(np.nanmean(np.roll(xa, s)[idx]))
            null = np.array(null)
            out[f"lag{k}"] = {"mean_lognp": float(obs), "null_mean": float(null.mean()),
                              "z": float((obs - null.mean()) / null.std()),
                              "p_high": float((1 + np.sum(null >= obs)) / (NN + 1))}
        res[f"chile_{sel_lab}_{per}"] = out
        print("CHILE", sel_lab, per, out, flush=True)

json.dump(res, open(out_json, "w"), indent=1, default=float)
