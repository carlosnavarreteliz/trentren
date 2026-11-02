"""Descriptive foreshock analysis: small-event activity before USGS M>=5 mainshocks vs matched control times.
Usage: python -I run_events.py <ipoc.txt>"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research"); sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog")
import numpy as np, pandas as pd, harness as H, ec_load, ec_features as F
from scipy import stats
OUT = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog/"
cat = ec_load.load_ipoc(sys.argv[1])
rng = np.random.default_rng(7)
u = H.get_events(min_mag=4.7)
box = (u.latitude > -24.5) & (u.latitude < -18.2) & (u.longitude > -71.8) & (u.longitude < -67.2)
u = u[box].copy()
tU, laU, loU = H.to_days(u.time), u.latitude.to_numpy(), u.longitude.to_numpy()
ms = u[(u.mag >= 5.0) & u.mainshock & (u.time >= "2008-01-01") & (u.time < "2022-01-01")].copy()
T0, T1 = H.to_days(pd.Series([pd.Timestamp("2008-01-01")]))[0], H.to_days(pd.Series([pd.Timestamp("2021-12-31")]))[0]
R_KM = 50.0
SEC = 60 / 86400

def hav(la1, lo1, la2, lo2):
    return H._haversine_km(la1, lo1, la2, lo2)

res = {}
for mthr in (3.0, 4.5):
    c = cat[cat.mag >= mthr]
    tc, lac, loc_, mc = H.to_days(c.time), c.latitude.to_numpy(), c.longitude.to_numpy(), c.mag.to_numpy()
    recs = []
    for _, e in ms.iterrows():
        near = hav(e.latitude, e.longitude, lac, loc_) <= R_KM
        tn = np.sort(tc[near]); mn = mc[near][np.argsort(tc[near])]
        nearU = hav(e.latitude, e.longitude, laU, loU)
        t_e = H.to_days(pd.Series([e.time]))[0]
        def counts(t):
            return [np.searchsorted(tn, t - SEC) - np.searchsorted(tn, t - d) for d in (1, 7, 30)]
        def quiet(t):  # no USGS M>=4.7 within 100 km in prior 30 d
            return not np.any((nearU <= 100) & (tU < t - SEC) & (tU >= t - 30))
        def bpos90(t):
            k0, k1 = np.searchsorted(tn, t - 90), np.searchsorted(tn, t - SEC)
            d = np.diff(mn[k0:k1]); d = d[d >= 0.2]
            return 1 / (np.log(10) * (d.mean() - 0.2)) if len(d) >= 15 else np.nan
        obs = counts(t_e)
        # controls: random times without a USGS M>=4.7 within 50 km in [t, t+7d)
        ctl = []
        while len(ctl) < 200:
            t = rng.uniform(T0 + 30, T1 - 7)
            if np.any((nearU <= R_KM) & (tU >= t) & (tU < t + 7)):
                continue
            ctl.append(t)
        ctl = np.array(ctl)
        C = np.array([counts(t) for t in ctl])
        Q = np.array([quiet(t) for t in ctl])
        bc = np.array([bpos90(t) for t in ctl[:100]])
        rec = dict(time=str(e.time), mag=float(e.mag), depth=float(e.depth), lat=float(e.latitude), quiet=bool(quiet(t_e)),
                   b90=float(bpos90(t_e)), b90_ctl_med=float(np.nanmedian(bc)) if np.isfinite(bc).any() else np.nan)
        for j, d in enumerate((1, 7, 30)):
            rec[f"n{d}"] = int(obs[j]); rec[f"ctl_mean{d}"] = float(C[:, j].mean())
            rec[f"exceed95_{d}"] = bool(obs[j] > np.quantile(C[:, j], 0.95))
            rec[f"pct{d}"] = float((np.sum(C[:, j] < obs[j]) + 0.5 * np.sum(C[:, j] == obs[j])) / len(C))
            cq = C[Q, j] if Q.sum() >= 30 else C[:, j]
            rec[f"exceed95q_{d}"] = bool(obs[j] > np.quantile(cq, 0.95))
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(OUT + f"events_foreshock_ML{mthr}.csv", index=False)
    summ = {}
    for sub, m in [("all", np.ones(len(df), bool)), ("shallow<=60", df.depth <= 60), ("deep>60", df.depth > 60),
                   ("quiet_history", df.quiet), ("M>=6", df.mag >= 6)]:
        d = df[m]
        s = {"n": int(len(d))}
        for w in (1, 7, 30):
            col = f"exceed95q_{w}" if sub == "quiet_history" else f"exceed95_{w}"
            k = int(d[col].sum())
            s[f"frac_exceed95_{w}d"] = round(k / max(len(d), 1), 3)
            s[f"binom_p_{w}d"] = float(stats.binomtest(k, len(d), 0.05, alternative="greater").pvalue) if len(d) else np.nan
            s[f"ratio_obs_ctl_{w}d"] = round(float(d[f"n{w}"].sum() / max(d[f"ctl_mean{w}"].sum(), 1e-9)), 2)
            s[f"mean_pct_{w}d"] = round(float(d[f"pct{w}"].mean()), 3)
        bb = d[["b90", "b90_ctl_med"]].dropna()
        s["b90_minus_ctl_median"] = round(float((bb.b90 - bb.b90_ctl_med).median()), 3) if len(bb) else None
        s["b90_wilcoxon_p"] = float(stats.wilcoxon(bb.b90 - bb.b90_ctl_med).pvalue) if len(bb) > 10 else None
        summ[sub] = s
    res[f"ML>={mthr}"] = summ
    print(f"ML>={mthr}", json.dumps(summ, indent=1))
    print(df[df.mag >= 6][["time", "mag", "depth", "quiet", "n1", "ctl_mean1", "n7", "ctl_mean7", "n30", "ctl_mean30", "pct7", "b90", "b90_ctl_med"]].to_string())
json.dump(res, open(OUT + "events_foreshock.json", "w"), indent=1, default=float)
