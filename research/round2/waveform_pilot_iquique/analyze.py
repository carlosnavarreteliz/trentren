"""Preregistered descriptive comparison of the enhanced catalogs: target T2014 vs controls C2013, C2017.

Usage: /opt/miniconda3/bin/python3 -I analyze.py <picks_dir> <assoc_dir> <meas_dir> <catalog_csv> <out_dir>
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pdir, adir, mdir, catf, odir = map(Path, sys.argv[1:6]); odir.mkdir(parents=True, exist_ok=True)
WINS = {"T2014": 2014, "C2013": 2013, "C2017": 2017}
REF = (-19.610, -70.769)
R_BOX = dict(lat=(-21.0, -18.5), lon=(-71.6, -69.8), dmax=60.0)
rng = np.random.default_rng(0)


def win_bounds(y):
    return pd.Timestamp(f"{y-1}-12-22"), pd.Timestamp(f"{y}-03-16 21:00"), pd.Timestamp(f"{y}-04-01 23:46")


def epi(lat, lon, la, lo):
    dx = (lon - lo) * 111.19 * np.cos(np.radians(0.5 * (lat + la))); dy = (lat - la) * 111.19
    return np.sqrt(dx ** 2 + dy ** 2)


# ---------- ComCat ----------
cat = pd.read_csv(catf)
cat["time"] = pd.to_datetime(cat.time, format="mixed").dt.tz_localize(None)
cat = cat[(cat.type == "earthquake") & cat.latitude.between(-22.5, -17.5) & cat.longitude.between(-72.5, -68.0)]

# ---------- events + raw magnitudes ----------
E = {}
for w in WINS:
    ev = pd.read_parquet(adir / f"events_{w}.parquet"); ev = ev[ev.n_sta >= 3].copy()
    am = pd.read_parquet(mdir / f"amps_{w}.parquet")
    am = am[am.A_mm > 0]
    am["ml"] = np.log10(am.A_mm) + 1.11 * np.log10(am.R_km) + 0.00189 * am.R_km + 0.591  # IASPEI form, A as WA amplitude in mm
    mlr = am.groupby("event_id").ml.median().rename("ml_raw")
    ev = ev.merge(mlr, left_on="event_id", right_index=True, how="left")
    E[w] = ev


def match(ev, cw):
    """match ComCat events to enhanced events: |dt|<=10 s and epicentral <= 120 km, nearest in time."""
    out = []
    for _, c in cw.iterrows():
        dt = (ev.origin - c.time).dt.total_seconds().abs()
        cand = ev[(dt <= 10)]
        if len(cand) == 0:
            out.append((c.time, c.mag, np.nan, np.nan, np.nan)); continue
        d = epi(cand.latitude.values, cand.longitude.values, c.latitude, c.longitude)
        ok = d <= 120
        if not ok.any():
            out.append((c.time, c.mag, np.nan, np.nan, np.nan)); continue
        j = np.argmin(np.where(ok, dt.loc[cand.index].values, 1e9))
        out.append((c.time, c.mag, cand.ml_raw.values[j], d[j], c.depth))
    return pd.DataFrame(out, columns=["time", "mag", "ml_raw", "dist", "depth"])


cal = {}
for w, y in WINS.items():
    t0, tp, t1 = win_bounds(y)
    cw = cat[(cat.time >= t0) & (cat.time < t1)]
    m = match(E[w], cw)
    cal[w] = dict(n_comcat=len(cw), n_matched=int(m.ml_raw.notna().sum()),
                  offset=float(np.nanmedian(m.mag - m.ml_raw)) if m.ml_raw.notna().any() else np.nan,
                  recall=float(m.ml_raw.notna().mean()) if len(m) else np.nan)
OFFSET = cal["C2013"]["offset"]  # preregistered: calibrate on C2013 only
for w in WINS:
    E[w]["mag"] = E[w].ml_raw + OFFSET

# ---------- uptime ----------
UP = {}
for w in WINS:
    u = pd.concat([pd.read_csv(f) for f in pdir.glob(f"uptime_{w}_*.csv")])
    piv = u.pivot_table(index="day", columns="station", values="uptime").reindex(columns=["PSGCX", "PB11", "PATCX"])
    UP[w] = piv.fillna(0)
valid_days = {w: set(UP[w].index[(UP[w] >= 0.9).all(axis=1)]) for w in WINS}


def in_R(ev):
    return (ev.latitude.between(*R_BOX["lat"]) & ev.longitude.between(*R_BOX["lon"]) & (ev.depth <= R_BOX["dmax"]))


def mc_maxc(m):
    m = m.dropna()
    if len(m) < 50:
        return np.nan
    bins = np.round(m, 1)
    vc = bins.value_counts().sort_index()
    return float(vc.idxmax()) + 0.2


def bval(m, mc, nboot=300):
    m = m[m >= mc - 0.05]
    if len(m) < 30:
        return np.nan, np.nan, len(m)
    b = np.log10(np.e) / (m.mean() - (mc - 0.05))
    bs = [np.log10(np.e) / (x.mean() - (mc - 0.05)) for x in (rng.choice(m.values, len(m)) for _ in range(nboot))]
    return float(b), float(np.std(bs)), int(len(m))


RE = {w: E[w][in_R(E[w])].copy() for w in WINS}
mcs = {w: mc_maxc(RE[w].mag) for w in WINS}
MC = float(np.nanmax(list(mcs.values())))


def ndays_valid(w, a, b):
    days = pd.date_range(a.normalize(), b.normalize(), freq="D")
    tot = 0.0
    for d in days:
        if d.strftime("%Y-%m-%d") in valid_days[w]:
            lo, hi = max(a, d), min(b, d + pd.Timedelta(days=1))
            tot += max(0, (hi - lo).total_seconds()) / 86400
    return tot


def valid_mask(w, t):
    return t.dt.strftime("%Y-%m-%d").isin(valid_days[w])


# ---------- families (S5) ----------
def families(w, ev):
    z = np.load(mdir / f"snip_{w}.npz")
    ids, W = z["ids"], z["w"]
    idx = {k: i for i, k in enumerate(ids)}
    sel = ev[ev.event_id.isin(idx)]
    if len(sel) < 2:
        return dict(n_ev=len(sel), frac_in_fam=np.nan, n_fam=0)
    X = W[[idx[k] for k in sel.event_id]].astype(np.float64)
    X -= X.mean(1, keepdims=True); X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    nf = 512; F = np.fft.rfft(X, nf)
    la, lo = sel.latitude.values, sel.longitude.values
    parent = list(range(len(sel)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(len(sel) - 1):
        j = np.arange(i + 1, len(sel))
        j = j[epi(la[j], lo[j], la[i], lo[i]) <= 30]
        if len(j) == 0:
            continue
        cc = np.fft.irfft(F[j] * np.conj(F[i])[None, :], nf)
        lagc = np.concatenate([cc[:, :21], cc[:, -20:]], axis=1).max(1)
        for jj in j[lagc >= 0.90]:
            ra, rb = find(i), find(jj)
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(i) for i in range(len(sel))])
    vc = pd.Series(roots).value_counts()
    fam = vc[vc >= 2]
    return dict(n_ev=int(len(sel)), frac_in_fam=float(fam.sum() / len(sel)), n_fam=int(len(fam)),
                largest=int(fam.max()) if len(fam) else 0)


# ---------- tremor-like (S6) ----------
def tremor(w, ev_all, a, b):
    env = pd.read_parquet(mdir / f"env_{w}.parquet")
    piv = env.pivot_table(index="hour", columns="station", values="env")
    piv = piv[(piv.index >= a) & (piv.index < b)]
    rel = piv / piv.median()
    hot = (rel > 3).all(axis=1) & piv.notna().all(axis=1)
    evh = set(ev_all.origin.dt.floor("h"))
    hot_noev = hot & ~pd.Series(piv.index.isin(list(evh)), index=piv.index)
    nvh = piv.notna().all(axis=1).sum()
    return dict(hot_hours=int(hot.sum()), hot_hours_no_event=int(hot_noev.sum()), valid_hours=int(nvh),
                per_valid_day=float(hot_noev.sum() / max(nvh, 1) * 24))


# ---------- statistics ----------
res = {"Mc_per_window": mcs, "Mc_common": MC, "calibration": cal, "offset_used": OFFSET, "windows": {}}
for w, y in WINS.items():
    t0, tp, t1 = win_bounds(y)
    ev = RE[w]; evc = ev[(ev.mag >= MC) & valid_mask(w, ev.origin)]
    out = {"n_events_all_assoc": int(len(E[w])), "n_events_R": int(len(ev)),
           "valid_days": len(valid_days[w]),
           "uptime_mean": UP[w].mean().round(3).to_dict()}
    for cut_name, tc in (("PRE", tp), ("FULL", t1)):
        e = evc[(evc.origin >= t0) & (evc.origin < tc)]
        nd = ndays_valid(w, t0, tc)
        rate = len(e) / nd if nd > 0 else np.nan
        tl = tc - pd.Timedelta(days=30)
        e_last, e_early = e[e.origin >= tl], e[e.origin < tl]
        nd_last, nd_early = ndays_valid(w, tl, tc), ndays_valid(w, t0, tl)
        r_last = len(e_last) / nd_last if nd_last else np.nan
        r_early = len(e_early) / nd_early if nd_early else np.nan
        dist = epi(e.latitude.values, e.longitude.values, *REF)
        rho = spearmanr((e.origin - t0).dt.total_seconds(), dist)[0] if len(e) > 10 else np.nan
        dmed_shift = (float(np.median(epi(e_last.latitude.values, e_last.longitude.values, *REF))
                            - np.median(epi(e_early.latitude.values, e_early.longitude.values, *REF)))
                      if len(e_last) > 5 and len(e_early) > 5 else np.nan)
        b_all = bval(ev[(ev.origin >= t0) & (ev.origin < tc)].mag.dropna(), MC)
        b_last = bval(ev[(ev.origin >= tl) & (ev.origin < tc)].mag.dropna(), MC)
        cc = cat[(cat.time >= t0) & (cat.time < tc) & cat.latitude.between(*R_BOX["lat"]) &
                 cat.longitude.between(*R_BOX["lon"]) & (cat.depth <= R_BOX["dmax"])]
        out[cut_name] = dict(
            S1_rate_per_day=rate, n=int(len(e)), valid_days=nd,
            S2_accel_ratio=(r_last / r_early) if r_early else np.nan, rate_last30=r_last, rate_early=r_early,
            S3_spearman_time_dist=float(rho), S3_median_dist_shift_km=dmed_shift,
            S3_median_dist_km=float(np.median(dist)) if len(dist) else np.nan,
            S4_b=b_all[0], S4_b_se=b_all[1], S4_b_n=b_all[2],
            S4_b_last30=b_last[0], S4_b_last30_se=b_last[1], S4_b_last30_n=b_last[2],
            S5=families(w, e), S6=tremor(w, E[w], t0, tc),
            S0_comcat_n_R=int(len(cc)), S0_comcat_rate_per_day=len(cc) / ((tc - t0).total_seconds() / 86400),
            S0_enh_over_comcat=(len(e) / len(cc)) if len(cc) else np.nan)
    # 10-day block rates (dispersion), PRE period
    blocks = []
    b0 = t0
    while b0 + pd.Timedelta(days=10) <= tp:
        b1 = b0 + pd.Timedelta(days=10)
        nd = ndays_valid(w, b0, b1)
        if nd >= 5:
            blocks.append(len(evc[(evc.origin >= b0) & (evc.origin < b1)]) / nd)
        b0 = b1
    out["block10_rates_PRE"] = blocks
    res["windows"][w] = out

# decision rule
dec = {}
def getv(w, key, sub=None, cut="PRE"):
    v = res["windows"][w][cut][key]
    return v[sub] if sub else v
for key, sub in [("S1_rate_per_day", None), ("S2_accel_ratio", None), ("S3_spearman_time_dist", None),
                 ("S3_median_dist_shift_km", None), ("S4_b", None), ("S4_b_last30", None),
                 ("S5", "frac_in_fam"), ("S6", "per_valid_day"), ("S0_enh_over_comcat", None)]:
    for cut in ("PRE", "FULL"):
        T = getv("T2014", key, sub, cut); C = [getv(c, key, sub, cut) for c in ("C2013", "C2017")]
        if any(pd.isna([T] + C)):
            dec[f"{key}{'.'+sub if sub else ''}|{cut}"] = dict(T=T, C2013=C[0], C2017=C[1], distinguishable=None); continue
        spread = abs(C[0] - C[1]); outside = T > max(C) or T < min(C)
        gap = min(abs(T - C[0]), abs(T - C[1]))
        dec[f"{key}{'.'+sub if sub else ''}|{cut}"] = dict(T=T, C2013=C[0], C2017=C[1], spread=spread, gap=gap,
                                                           distinguishable=bool(outside and gap > spread))
res["decision"] = dec
# dispersion check on S1 PRE vs control 10-day blocks
cb = np.array(res["windows"]["C2013"]["block10_rates_PRE"] + res["windows"]["C2017"]["block10_rates_PRE"])
T1 = res["windows"]["T2014"]["PRE"]["S1_rate_per_day"]
res["S1_vs_control_blocks"] = dict(z=float((T1 - cb.mean()) / cb.std(ddof=1)), rank_frac=float((cb >= T1).mean()),
                                   n_blocks=int(len(cb)), control_block_mean=float(cb.mean()),
                                   control_block_sd=float(cb.std(ddof=1)))
tb = np.array(res["windows"]["T2014"]["block10_rates_PRE"])
res["T_blocks_vs_control_max"] = dict(T_blocks=tb.tolist(), control_max=float(cb.max()),
                                      n_T_blocks_above_control_max=int((tb > cb.max()).sum()))
# ComCat contrast vs enhanced contrast (PRE)
for c in ("C2013", "C2017"):
    enh = T1 / res["windows"][c]["PRE"]["S1_rate_per_day"]
    cc = res["windows"]["T2014"]["PRE"]["S0_comcat_rate_per_day"] / max(res["windows"][c]["PRE"]["S0_comcat_rate_per_day"], 1e-9)
    res[f"contrast_T_over_{c}"] = dict(enhanced=float(enh), comcat_M4=float(cc))

json.dump(res, open(odir / "results.json", "w"), indent=1, default=lambda o: None if o is None else float(o))

# ---------- figures + daily series for dashboard ----------
fig, ax = plt.subplots(4, 1, figsize=(10, 13))
daily = {}
for w, y in WINS.items():
    t0, tp, t1 = win_bounds(y)
    e = RE[w][(RE[w].mag >= MC)]
    dd = (e.origin - t0).dt.total_seconds() / 86400
    cnt = np.histogram(dd, bins=np.arange(0, 102))[0].astype(float)
    vd = np.array([(t0 + pd.Timedelta(days=i)).strftime("%Y-%m-%d") in valid_days[w] for i in range(101)])
    cnt[~vd] = np.nan
    daily[w] = cnt.tolist()
    ax[0].plot(np.arange(101), cnt, label=w, lw=1)
    ax[1].plot(np.sort(dd), np.arange(1, len(dd) + 1), label=w)
    ax[2].scatter(dd, epi(e.latitude.values, e.longitude.values, *REF), s=3, label=w, alpha=.5)
    ax[3].scatter(dd, e.mag, s=3, label=w, alpha=.5)
for a in ax:
    a.axvline((pd.Timestamp("2014-03-16 21:00") - pd.Timestamp("2013-12-22")).days, color="k", ls="--", lw=.8)
    a.legend(); a.set_xlabel("days since Dec 22")
ax[0].set_ylabel(f"events/day R, M>={MC:.1f}"); ax[1].set_ylabel("cumulative"); ax[2].set_ylabel("dist to 2014 hypo (km)")
ax[3].set_ylabel("M proxy")
plt.tight_layout(); plt.savefig(odir / "fig_windows.png", dpi=110)
json.dump(dict(daily_counts_R=daily, Mc=MC), open(odir / "daily_series.json", "w"))
print(json.dumps({k: res[k] for k in ("Mc_per_window", "Mc_common", "calibration", "S1_vs_control_blocks",
                                      "T_blocks_vs_control_max")}, indent=1, default=str))
for k, v in dec.items():
    print(k, v)
for w in WINS:
    print(w, {k: v for k, v in res["windows"][w].items() if k not in ("PRE", "FULL", "block10_rates_PRE")})
    for cut in ("PRE", "FULL"):
        print("  ", cut, res["windows"][w][cut])
