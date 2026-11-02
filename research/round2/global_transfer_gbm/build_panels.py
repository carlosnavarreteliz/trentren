"""Build harness-equivalent causal feature panels for every subduction margin + Chile.

Usage: python3 -I build_panels.py <raw_dir> <out_dir> [target_mag ...]
  raw_dir: folder with global_m47_<year>.csv (USGS FDSN downloads, untrusted -> only parsed by pandas)
  out_dir: writes events_margins.parquet, panel_<CODE>_m<M>.parquet, margins_info.json
"""
import sys, os, json, glob, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import harness as H
import margins as MG

RAW, OUT = sys.argv[1], sys.argv[2]
TMAGS = [float(x) for x in sys.argv[3:]] or [6.0, 5.5]
os.makedirs(OUT, exist_ok=True)
STEP, HOR = 7, 7


def load_global(raw):
    fs = sorted(glob.glob(os.path.join(raw, "global_m47_*.csv")))
    d = pd.concat([pd.read_csv(f, usecols=["time", "latitude", "longitude", "depth", "mag", "magType", "id", "type"])
                   for f in fs], ignore_index=True)
    d = d[d["type"] == "earthquake"].dropna(subset=["mag", "latitude", "longitude"]).copy()
    d["time"] = pd.to_datetime(d["time"], format="ISO8601", utc=True).dt.tz_convert(None)
    d = d[(d.mag >= H.M_MODEL) & (d.time >= H.HISTORY_START)].drop_duplicates("id")
    d["depth"] = d["depth"].fillna(33.0)
    d = d.sort_values("time").reset_index(drop=True)
    d["t_days"] = H.to_days(d["time"])
    return d


def bvalue_feature(ti, ev, nb, nbr=1, last_n=100, window=3652.5, min_n=30):
    out = np.full((len(ti), nb), np.nan)
    bins, te, me = ev.bin_id.to_numpy(), ev.t_days.to_numpy(), ev.mag.to_numpy()
    for b in range(nb):
        sel = np.abs(bins - b) <= nbr
        o = np.argsort(te[sel])
        tb, mb = te[sel][o], me[sel][o]
        cs = np.r_[0, np.cumsum(mb)]
        hi = np.searchsorted(tb, ti, side="left")
        lo = np.maximum(np.searchsorted(tb, ti - window, side="left"), hi - last_n)
        n = hi - lo
        mean = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        bv = np.log10(np.e) / np.maximum(mean - (H.M_MODEL - 0.05), 0.05)
        out[:, b] = np.where(n >= min_n, bv, np.nan)
    return out


def build_margin_panel(ev, edges, t_starts, target_mags):
    """ev: events with t_days, mag, latitude (= along-strike coordinate in deg), bin_id, mainshock, time.
    Returns {target_mag: DataFrame} with the harness baseline columns + extras + targets."""
    nb = len(edges) - 1
    ti = H.to_days(t_starts)
    h = float(HOR)
    feats = H._etas_features(ti, ev, edges, h)
    feats.update(H._count_features(ti, ev, nb, [7, 30, 365], 0, "cnt_bin"))
    feats.update(H._count_features(ti, ev, nb, [30, 365], 1, "cnt_nbr"))
    feats.update(H._maxmag_features(ti, ev, nb, [30, 365]))
    feats["log_days_since_m6_nbr"] = np.log10(1 + H._days_since(ti, ev, nb, 6.0))
    # extras (F_ext)
    ex = {}
    c2 = H._count_features(ti, ev, nb, [7, 3650], 1, "cnt_nbr")
    ex["xx_cnt_nbr_7d"], ex["xx_cnt_nbr_3650d"] = c2["cnt_nbr_7d"], c2["cnt_nbr_3650d"]
    m2 = H._maxmag_features(ti, ev, nb, [7, 3650])
    ex["xx_maxmag_nbr_7d"], ex["xx_maxmag_nbr_3650d"] = m2["maxmag_nbr_7d"], m2["maxmag_nbr_3650d"]
    ex["xx_log_days_since_m7_nbr"] = np.log10(1 + H._days_since(ti, ev, nb, 7.0))
    ex["xx_bvalue"] = bvalue_feature(ti, ev, nb)

    T, B = np.meshgrid(np.arange(len(ti)), np.arange(nb), indexing="ij")
    base = pd.DataFrame({"bin_id": B.ravel().astype(np.int16), "lat_center": 0.5 * (edges[:-1] + edges[1:])[B.ravel()],
                         "t_start": t_starts[T.ravel()]})
    base["year"] = base.t_start.dt.year.astype(np.int16)
    end_t = base.t_start + pd.Timedelta(days=HOR)
    base["split"] = np.where(base.t_start >= H.TEST_START, "test", np.where(end_t <= H.TEST_START, "train", "gap"))
    out = {}
    ev_tr = ev[(ev.time >= H.PANEL_START) & (ev.time < H.TEST_START)]
    span_days = (H.TEST_START - H.PANEL_START).days
    rate45 = np.bincount(ev_tr.bin_id[ev_tr.bin_id >= 0], minlength=nb)[:nb] / span_days * h
    for tm in target_mags:
        p = base.copy()
        for name, e in [("all", ev[ev.mag >= tm]), ("main", ev[(ev.mag >= tm) & ev.mainshock])]:
            cnt = np.zeros((len(ti), nb), int)
            for b in range(nb):
                tb = np.sort(e.t_days.to_numpy()[e.bin_id.to_numpy() == b])
                cnt[:, b] = np.searchsorted(tb, ti + h, side="left") - np.searchsorted(tb, ti, side="left")
            p[f"y_{name}"] = (cnt.ravel() > 0).astype(np.int8)
            p[f"n_{name}"] = cnt.ravel()
        for k, v in feats.items():
            p[k] = v.ravel().astype(np.float64)
        tr = p.split == "train"
        p["log_bg_m45"] = np.log10(H._smooth_bins(rate45.astype(float), 1.0) + 1e-4)[p.bin_id]
        for tgt in ["all", "main"]:
            k = p[tr].groupby("bin_id")[f"y_{tgt}"].agg(["sum", "count"]).reindex(range(nb), fill_value=0)
            glob_ = k["sum"].sum() / k["count"].sum()
            freq = (H._smooth_bins(k["sum"].to_numpy().astype(float), 1.0) + glob_) / (k["count"].to_numpy() + 1.0)
            p[f"p_poisson_{tgt}"] = np.clip(freq, 1e-6, 0.5)[p.bin_id]
        p["log_bg_target"] = np.log10(p["p_poisson_all"])
        for k, v in ex.items():
            p[k] = v.ravel().astype(np.float64)
        out[tm] = p
    return out


def main():
    t0 = time.time()
    hp = H.get_panel(step_days=STEP, horizon_days=HOR, target_mag=6.0)
    t_starts = pd.DatetimeIndex(np.sort(hp.t_start.unique()))
    info = {"t_first": str(t_starts[0]), "t_last": str(t_starts[-1]), "margins": {}}

    # ---------------- Chile: harness events, harness declustering, latitude bins ----------------
    evc = H.get_events(min_mag=H.M_MODEL)
    eta0 = float(H._DECL["zbz"].attrs.get("log10_eta0", np.nan))
    info["log10_eta0_chile"] = eta0
    edges_c = np.arange(H.LAT_MIN, H.LAT_MAX + 1e-9, 1.0)
    pc = build_margin_panel(evc, edges_c, t_starts, TMAGS)
    # check reproduction of the harness panel
    chk = {}
    for tm in TMAGS:
        hpm = H.get_panel(step_days=STEP, horizon_days=HOR, target_mag=tm)
        cols = H.baseline_feature_cols(hpm) + ["y_all", "y_main", "p_poisson_all", "p_poisson_main"]
        a = pc[tm][cols].to_numpy(float)
        b = hpm[cols].to_numpy(float)
        chk[str(tm)] = {"max_abs_diff": float(np.nanmax(np.abs(a - b))), "rows": [len(a), len(b)]}
        pc[tm].to_parquet(os.path.join(OUT, f"panel_CHL_m{tm}.parquet"), index=False)
    info["chile_reproduction"] = chk
    info["margins"]["CHL"] = {"n_bins": len(edges_c) - 1, "n_events": int(len(evc)),
                              "n_m6": int((evc.mag >= 6).sum()), "n_m7": int((evc.mag >= 7).sum())}
    print("Chile done", chk, round(time.time() - t0, 1), flush=True)

    # ---------------- global margins ----------------
    g = load_global(RAW)
    info["n_global_events"] = int(len(g))
    code, s_km, d_km, signs = MG.assign(g.longitude.to_numpy(), g.latitude.to_numpy(), g.depth.to_numpy())
    g["margin"], g["s_km"], g["d_km"] = code, s_km, d_km
    info["landward_sign"] = {k: float(v) for k, v in signs.items()}
    # how much of the Chile box in the global download matches the harness catalog
    chl = g[MG.in_chile_box(g.longitude.to_numpy(), g.latitude.to_numpy())]
    info["chile_box_global_vs_harness"] = [int(len(chl)), int(len(evc))]
    gm = g[g.margin != ""].copy()
    gm.drop(columns=["type"]).to_parquet(os.path.join(OUT, "events_margins.parquet"), index=False)
    for m, poly in MG.MARGINS.items():
        e = gm[gm.margin == m].copy().reset_index(drop=True)
        L = float(MG.polyline_length_km(poly))
        nb = int(np.ceil(L / MG.KM_PER_DEG))
        e["latitude_geo"] = e["latitude"]
        dec = H.decluster(cat=e, method="zbz", eta0=eta0)       # uses geographic lat/lon for distances
        dec["latitude"] = dec["s_km"] / MG.KM_PER_DEG              # along-strike coordinate (deg) for features
        dec["bin_id"] = np.minimum(np.floor(dec["latitude"]).astype(int), nb - 1)
        edges = np.arange(0, nb + 1e-9, 1.0)
        pm = build_margin_panel(dec, edges, t_starts, TMAGS)
        for tm in TMAGS:
            pm[tm].to_parquet(os.path.join(OUT, f"panel_{m}_m{tm}.parquet"), index=False)
        info["margins"][m] = {"n_bins": nb, "length_km": round(L, 1), "n_events": int(len(dec)),
                              "n_m6": int((dec.mag >= 6).sum()), "n_m7": int((dec.mag >= 7).sum()),
                              "n_m8": int((dec.mag >= 8).sum()),
                              "n_m6_main": int(((dec.mag >= 6) & dec.mainshock).sum()),
                              "test_pos_m6_all": int(pm[6.0].query("split=='test'").y_all.sum()) if 6.0 in pm else None,
                              "test_pos_m6_main": int(pm[6.0].query("split=='test'").y_main.sum()) if 6.0 in pm else None}
        print(m, info["margins"][m], round(time.time() - t0, 1), flush=True)
    json.dump(info, open(os.path.join(OUT, "margins_info.json"), "w"), indent=1)
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
