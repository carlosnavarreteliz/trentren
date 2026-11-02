"""Build the pretraining data set from global ComCat (1973-2012) on 10 non-Chile subduction margins, with features
defined exactly as the harness Chile panel, plus the Chile arrays for the network.

Usage: python3 -I build_global.py COMCAT_DIR OUT_DIR
Outputs (OUT_DIR):
  global_events.parquet        events assigned to margins (s_km, d_km, margin, bin_id, pseudo-lat)
  global_samples.npz           tabular X (raw harness-style, transformed), targets Y (5 heads), keys, sequence index
  chile_samples.npz            same for the Chile panel (all rows incl. test; split flags kept)
"""
import sys, os, glob, time, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H

COMCAT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
KM_PER_DEG = 111.2
SPLIT = H.TEST_START
D_MIN, D_MAX = -200.0, 700.0
SEQ_LEN, NBR = 52, 2

# --- trench polylines (lon, lat) and a landward reference point --------------------------------------------
MARGINS = {
    "peru_ecuador_colombia": ([(-72.0, -17.6), (-74.0, -16.0), (-76.3, -14.5), (-77.8, -13.0), (-78.8, -11.5),
                               (-80.0, -9.5), (-81.0, -7.5), (-81.7, -5.5), (-81.6, -3.5), (-81.0, -1.5),
                               (-80.4, 0.5), (-79.5, 2.0), (-78.6, 3.5), (-78.0, 5.5)], (-77.03, -12.05)),
    "mexico_central_america": ([(-106.0, 20.5), (-104.5, 18.5), (-102.5, 17.4), (-100.0, 16.2), (-98.0, 15.4),
                                (-95.5, 14.6), (-93.5, 13.9), (-91.5, 13.0), (-89.5, 12.2), (-87.5, 11.2),
                                (-86.0, 10.2), (-85.0, 9.0), (-84.0, 7.8), (-82.8, 7.0)], (-99.13, 19.43)),
    "japan_kuril_kamchatka": ([(141.9, 34.0), (142.8, 35.5), (143.7, 37.0), (144.2, 38.5), (144.4, 40.0),
                               (144.6, 41.0), (145.6, 41.8), (147.5, 42.6), (149.5, 43.6), (151.5, 44.9),
                               (153.5, 46.3), (155.5, 47.9), (157.5, 49.6), (159.0, 51.0), (160.5, 52.5),
                               (162.0, 54.0), (163.5, 55.6), (164.5, 56.5)], (140.87, 38.27)),
    "aleutians_alaska": ([(164.5, 54.2), (168.0, 52.4), (172.0, 51.0), (176.0, 50.4), (180.0, 50.4),
                          (-176.0, 50.6), (-172.0, 51.0), (-168.0, 51.8), (-164.0, 52.9), (-160.0, 54.0),
                          (-156.0, 55.2), (-152.0, 56.7), (-149.0, 58.2), (-146.5, 59.3)], (-149.9, 61.2)),
    "izu_bonin_mariana": ([(142.0, 34.0), (142.0, 32.0), (142.2, 30.0), (142.6, 28.0), (143.1, 26.0),
                           (143.6, 24.0), (144.5, 22.0), (146.0, 20.0), (147.3, 18.0), (147.7, 16.0),
                           (147.2, 14.0), (146.0, 12.3), (144.0, 11.3), (142.0, 11.0), (140.0, 10.5)],
                          (144.79, 13.44)),
    "nankai_ryukyu": ([(138.5, 34.0), (137.0, 33.5), (135.5, 32.9), (134.0, 32.4), (132.5, 31.5), (131.6, 30.0),
                       (130.6, 28.5), (129.4, 27.0), (128.0, 25.6), (126.5, 24.5), (125.0, 23.8), (123.5, 23.4),
                       (122.5, 23.3)], (135.5, 34.7)),
    "tonga_kermadec": ([(-172.5, -15.0), (-173.0, -16.5), (-173.2, -18.5), (-173.3, -20.5), (-174.2, -22.5),
                        (-175.3, -24.5), (-176.0, -26.5), (-176.6, -28.5), (-177.1, -30.5), (-177.6, -32.5),
                        (-178.2, -34.5), (-178.9, -36.5), (-179.8, -38.0), (178.8, -39.5)], (-175.2, -21.13)),
    "sunda": ([(91.9, 14.5), (91.9, 12.0), (92.0, 10.0), (92.6, 8.0), (93.8, 6.0), (95.0, 4.0), (96.3, 2.0),
               (97.6, 0.0), (99.0, -2.0), (100.6, -4.0), (102.4, -6.0), (104.6, -7.8), (107.0, -9.3),
               (110.0, -10.2), (113.0, -10.8), (116.0, -11.2), (119.0, -11.3), (121.0, -11.0)], (100.35, -0.95)),
    "vanuatu": ([(165.5, -10.0), (165.8, -12.0), (166.2, -14.0), (166.6, -16.0), (167.3, -18.0), (168.3, -20.0),
                 (169.6, -21.8), (171.0, -22.8), (172.5, -23.0)], (168.32, -17.73)),
    "philippine_trench": ([(126.5, 14.0), (126.8, 12.0), (127.0, 10.0), (127.1, 8.0), (127.0, 6.5), (126.6, 5.0),
                           (126.0, 3.5)], (125.6, 7.07)),
}


def unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def densify(pts, step_km=2.0):
    P = unit(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        ang = np.arccos(np.clip(a @ b, -1, 1))
        n = max(1, int(np.ceil(ang * 6371 / step_km)))
        for k in range(1, n + 1):  # slerp
            f = k / n
            v = (np.sin((1 - f) * ang) * a + np.sin(f * ang) * b) / np.sin(ang)
            out.append(v)
    D = np.array(out)
    seg = np.arccos(np.clip(np.sum(D[1:] * D[:-1], 1), -1, 1)) * 6371
    s = np.r_[0, np.cumsum(seg)]
    T = np.gradient(D, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return D, s, T


def project(lon, lat, pts, land):
    D, s, T = densify(pts)
    Q = unit(lon, lat)
    out_s = np.empty(len(Q)); out_d = np.empty(len(Q)); inner = np.empty(len(Q), bool)
    for c in range(0, len(Q), 20000):
        q = Q[c:c + 20000]
        dots = q @ D.T
        i = np.argmax(dots, 1)
        ang = np.arccos(np.clip(dots[np.arange(len(q)), i], -1, 1)) * 6371
        side = np.sign(np.einsum("ij,ij->i", np.cross(D[i], T[i]), q))
        out_s[c:c + 20000] = s[i]; out_d[c:c + 20000] = ang * side
        inner[c:c + 20000] = (i > 0) & (i < len(D) - 1)
    # landward sign from the reference point
    ql = unit(np.array([land[0]]), np.array([land[1]]))
    il = np.argmax(ql @ D.T)
    sgn = np.sign(np.cross(D[il], T[il]) @ ql[0])
    return out_s, out_d * sgn, inner, s[-1]


# --- load global catalog --------------------------------------------------------------------------------------
t0 = time.time()
files = sorted(glob.glob(os.path.join(COMCAT, "comcat_*.csv")))
cat = pd.concat([pd.read_csv(f, usecols=["time", "latitude", "longitude", "depth", "mag", "magType", "id", "type"])
                 for f in files], ignore_index=True)
cat = cat[cat["type"] == "earthquake"].drop_duplicates("id")
cat["time"] = pd.to_datetime(cat["time"], format="ISO8601", utc=True).dt.tz_convert(None)
cat = cat.dropna(subset=["mag", "latitude", "longitude"])
cat = cat[(cat.mag >= H.M_MODEL) & (cat.time >= H.HISTORY_START) & (cat.time < SPLIT)].sort_values("time")
in_chile = cat.latitude.between(H.LAT_MIN, H.LAT_MAX) & cat.longitude.between(H.LON_MIN, H.LON_MAX)
print(f"global events M>=4.7 1973-2012: {len(cat)}; inside Chile box (removed): {in_chile.sum()}", flush=True)
cat = cat[~in_chile].reset_index(drop=True)
assert cat.time.max() < SPLIT

lon, lat = cat.longitude.to_numpy(), cat.latitude.to_numpy()
best = np.full(len(cat), -1); best_abs = np.full(len(cat), np.inf)
S = {}; Dd = {}; L = {}
for k, (name, (pts, land)) in enumerate(MARGINS.items()):
    s, d, inner, length = project(lon, lat, pts, land)
    ok = inner & (d >= D_MIN) & (d <= D_MAX)
    better = ok & (np.abs(d) < best_abs)
    best[better] = k; best_abs[better] = np.abs(d)[better]
    S[k], Dd[k], L[name] = s, d, length
names = list(MARGINS)
cat["margin"] = best
cat = cat[cat.margin >= 0].copy()
cat["s_km"] = [S[m][i] for m, i in zip(cat.margin, cat.index)]
cat["d_km"] = [Dd[m][i] for m, i in zip(cat.margin, cat.index)]
cat["t_days"] = H.to_days(cat["time"])
cat["margin_name"] = [names[m] for m in cat.margin]
print("assigned:", cat.groupby("margin_name").size().to_dict(), "lengths km:", {k: round(v) for k, v in L.items()},
      flush=True)

# --- issue grid identical to the harness train rows -----------------------------------------------------------
H_DAYS = 7.0
t_starts = pd.date_range(H.PANEL_START, SPLIT - pd.Timedelta(days=7), freq="7D")
ti = H.to_days(t_starts)
# extended weekly grid for the sequences: g_j = PANEL_START + 7 j, j = -SEQ_LEN+1 .. len(ti)-1 ; window [g-7, g)
g_ext = H.to_days(H.PANEL_START) + 7.0 * np.arange(-(SEQ_LEN - 1), len(ti))


def weekly_stats(ev, nb, g):
    """[len(g), nb, 3]: log1p count M>=4.7, max(M-4.2) (0 none), log1p count M>=6 in [g-7, g)."""
    W = np.zeros((len(g), nb, 3), np.float32)
    te, b, m = ev.t_days.to_numpy(), ev.bin_id.to_numpy(), ev.mag.to_numpy()
    j = np.searchsorted(g, te, side="right")  # event at te belongs to window ending at first g > te
    ok = (j < len(g)) & (te >= g[0] - 7) & (b >= 0) & (b < nb)
    j, b, m = j[ok], b[ok], m[ok]
    np.add.at(W[:, :, 0], (j, b), 1.0)
    np.add.at(W[:, :, 2], (j, b), (m >= 6.0).astype(np.float32))
    np.maximum.at(W[:, :, 1], (j, b), (m - 4.2).astype(np.float32))
    W[:, :, 0] = np.log1p(W[:, :, 0]); W[:, :, 2] = np.log1p(W[:, :, 2])
    return W


def targets(ev, nb, ti, thr, main):
    e = ev[(ev.mag >= thr) & (ev.mainshock if main else True)]
    cnt = np.zeros((len(ti), nb), int)
    for b in range(nb):
        tb = np.sort(e.t_days.to_numpy()[e.bin_id.to_numpy() == b])
        cnt[:, b] = np.searchsorted(tb, ti + H_DAYS, side="left") - np.searchsorted(tb, ti, side="left")
    return (cnt > 0).astype(np.int8)


BASE_COLS = None
rows_X, rows_Y, rows_key, W_blocks = [], [], [], []
col_offset = 0
meta = {}
for k, name in enumerate(names):
    ev = cat[cat.margin == k].copy().sort_values("time").reset_index(drop=True)
    nb = int(np.ceil(L[name] / KM_PER_DEG))
    ev["bin_id"] = np.clip((ev.s_km // KM_PER_DEG).astype(int), 0, nb - 1)
    # declustering (ZBZ) on real coordinates -> mainshock flags (TARGET ONLY)
    dec = H.decluster(ev[["time", "t_days", "latitude", "longitude", "depth", "mag", "bin_id", "s_km"]].copy())
    ev = dec  # sorted by time
    ev["lat_real"] = ev["latitude"]
    ev_feat = ev.assign(latitude=ev.s_km / KM_PER_DEG)  # pseudo-latitude = along-strike degrees
    edges = np.arange(0, nb + 1, dtype=float)
    feats = H._etas_features(ti, ev_feat, edges, H_DAYS)
    feats.update(H._count_features(ti, ev_feat, nb, [7, 30, 365], 0, "cnt_bin"))
    feats.update(H._count_features(ti, ev_feat, nb, [30, 365], 1, "cnt_nbr"))
    feats.update(H._maxmag_features(ti, ev_feat, nb, [30, 365]))
    feats["log_days_since_m6_nbr"] = np.log10(1 + H._days_since(ti, ev_feat, nb, 6.0))
    Y = {"all55": targets(ev, nb, ti, 5.5, False), "main55": targets(ev, nb, ti, 5.5, True),
         "all6": targets(ev, nb, ti, 6.0, False), "main6": targets(ev, nb, ti, 6.0, True),
         "all7": targets(ev, nb, ti, 7.0, False)}
    # background rates (1976-2012), same formulas as harness.build_panel
    ev_tr = ev[(ev.time >= H.PANEL_START) & (ev.time < SPLIT)]
    span = (SPLIT - H.PANEL_START).days
    rate45 = np.bincount(ev_tr.bin_id, minlength=nb) / span * H_DAYS
    feats["log_bg_m45"] = np.repeat(np.log10(H._smooth_bins(rate45, 1.0) + 1e-4)[None, :], len(ti), 0)
    yk = Y["all6"]
    glob_ = yk.sum() / yk.size
    freq = (H._smooth_bins(yk.sum(0).astype(float), 1.0) + glob_) / (yk.shape[0] + 1.0)
    feats["log_bg_target"] = np.repeat(np.log10(np.clip(freq, 1e-6, 0.5))[None, :], len(ti), 0)
    cols = [c for c in H.baseline_feature_cols(pd.DataFrame(columns=list(feats)))]
    if BASE_COLS is None:
        BASE_COLS = cols
    df = pd.DataFrame({c: feats[c].ravel() for c in BASE_COLS})
    X = H._transform_base(df, BASE_COLS).astype(np.float32)
    T_, B_ = np.meshgrid(np.arange(len(ti)), np.arange(nb), indexing="ij")
    rows_X.append(X)
    rows_Y.append(np.stack([Y[h].ravel() for h in ["all55", "main55", "all6", "main6", "all7"]], 1))
    rows_key.append(np.stack([np.full(T_.size, k), T_.ravel(), B_.ravel(), col_offset + NBR + B_.ravel()], 1))
    W = weekly_stats(ev, nb, g_ext)
    Wp = np.zeros((len(g_ext), nb + 2 * NBR, 3), np.float32); Wp[:, NBR:NBR + nb] = W
    W_blocks.append(Wp); col_offset += nb + 2 * NBR
    n7 = int((ev.mag >= 7).sum())
    meta[name] = dict(n_events=len(ev), n_bins=nb, n_m6=int((ev.mag >= 6).sum()), n_m7=n7,
                      pos_all6=int(Y["all6"].sum()), pos_main6=int(Y["main6"].sum()), pos_all7=int(Y["all7"].sum()),
                      log10_eta0=float(dec.attrs.get("log10_eta0", np.nan)))
    print(name, meta[name], f"{time.time()-t0:.0f}s", flush=True)

np.savez_compressed(os.path.join(OUT, "global_samples.npz"), X=np.concatenate(rows_X), Y=np.concatenate(rows_Y),
                    key=np.concatenate(rows_key).astype(np.int32), W=np.concatenate(W_blocks, 1),
                    cols=np.array(BASE_COLS), t_starts=t_starts.values.astype("datetime64[ns]"))
cat.to_parquet(os.path.join(OUT, "global_events.parquet"), index=False)

# --- Chile arrays ------------------------------------------------------------------------------------------------
p6 = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
p55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
assert (p6[["bin_id", "t_start"]].values == p55[["bin_id", "t_start"]].values).all()
assert BASE_COLS == H.baseline_feature_cols(p6), (BASE_COLS, H.baseline_feature_cols(p6))
Xc = H._transform_base(p6, BASE_COLS).astype(np.float32)
evc = H.get_events(min_mag=4.7)
tc = H.to_days(p6.t_start.drop_duplicates().sort_values())
nbc = 29
p6t = H.to_days(p6.t_start)
# M>=7 target for Chile (all events)
e7 = evc[evc.mag >= 7.0]
y7 = np.zeros(len(p6), np.int8)
for b in range(nbc):
    tb = np.sort(e7.t_days.to_numpy()[e7.bin_id.to_numpy() == b])
    sel = (p6.bin_id.to_numpy() == b)
    tt = p6t[sel]
    y7[sel] = ((np.searchsorted(tb, tt + 7, side="left") - np.searchsorted(tb, tt, side="left")) > 0)
Yc = np.stack([p55.y_all, p55.y_main, p6.y_all, p6.y_main, y7], 1).astype(np.int8)
gc = tc[0] + 7.0 * np.arange(-(SEQ_LEN - 1), len(tc))
Wc = weekly_stats(evc, nbc, gc)
Wcp = np.zeros((len(gc), nbc + 2 * NBR, 3), np.float32); Wcp[:, NBR:NBR + nbc] = Wc
tidx = np.searchsorted(tc, p6t)
keyc = np.stack([np.full(len(p6), -1), tidx, p6.bin_id.to_numpy(), NBR + p6.bin_id.to_numpy()], 1).astype(np.int32)
np.savez_compressed(os.path.join(OUT, "chile_samples.npz"), X=Xc, Y=Yc, key=keyc, W=Wcp,
                    split=p6.split.to_numpy().astype(str), t_start=p6.t_start.values.astype("datetime64[ns]"),
                    bin_id=p6.bin_id.to_numpy())
meta["_chile"] = dict(rows=len(p6), train_pos=Yc[(p6.split == "train").to_numpy()].sum(0).tolist(),
                      heads=["all55", "main55", "all6", "main6", "all7"])
meta["_global_total"] = dict(rows=int(sum(len(x) for x in rows_X)),
                             pos=np.concatenate(rows_Y).sum(0).tolist(), n_events=len(cat))
json.dump(meta, open(os.path.join(OUT, "build_meta.json"), "w"), indent=1)
print(json.dumps(meta["_global_total"]), json.dumps(meta["_chile"]), f"{time.time()-t0:.0f}s")
