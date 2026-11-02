"""Hydroclimate hypothesis runner.
Usage: python -I run.py <data_dir> <stage> [n_surr]
stages: sanity | primary | secondary | robust | schuster
Writes results_<stage>.json next to this script."""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/hydroclimate")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import harness as H
import features as F

DDIR, STAGE = sys.argv[1], sys.argv[2]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
HERE = Path(__file__).parent
OUT = {}


def clean(d):
    return {k: v for k, v in d.items() if k not in ("preds", "null")}


def loyo(preds):
    """leave-one-test-year-out IG per event (full vs base)."""
    y = preds["y"].to_numpy(); pf, pb = np.clip(preds.p_full, 1e-12, 1 - 1e-12), np.clip(preds.p_base, 1e-12, 1 - 1e-12)
    ll = y * np.log(pf / pb) + (1 - y) * np.log((1 - pf) / (1 - pb))
    out = {}
    for yr in sorted(preds.year.unique()):
        m = (preds.year != yr).to_numpy()
        out[int(yr)] = float(ll[m].sum() / max(y[m].sum(), 1))
    return out


def ev(feats, target, panel, model="logit", tag=""):
    r = H.evaluate(feats, target=target, model=model, panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    lo = loyo(r["preds"])
    d = clean(r) | {"bootstrap": bb, "loyo_min": min(lo.values()), "loyo_max": max(lo.values()), "loyo": lo}
    print(f"[{tag}] {target} {model}: IG={r['ig_per_event']:+.4f} vsETAS={r['ig_full_vs_etas']:+.4f} "
          f"base_vsETAS={r['ig_base_vs_poisson']-r['ig_etas_vs_poisson']:+.4f} AUC {r['base']['roc_auc']:.3f}->{r['full']['roc_auc']:.3f} "
          f"ASS {r['base']['molchan_ass']:.3f}->{r['full']['molchan_ass']:.3f} boot={bb} loyo_min={min(lo.values()):+.4f} "
          f"coef={r.get('coef_extra', r.get('importance_extra'))}", flush=True)
    return d


def sur(feats=None, target="y_main", panel=None, n=NS, mode="phase_randomize", builder=None, tag="", seed=0):
    t0 = time.time()
    s = H.surrogate_test(feats, builder=builder, n=n, mode=mode, target=target, panel=panel, seed=seed, n_jobs=2)
    print(f"[{tag}] surrogate {mode} {target} n={n}: obs={s['ig_obs']:+.4f} p={s['p_value']:.4f} "
          f"null_mean={s['null_mean']:+.4f} q95={s['null_q95']:+.4f} ({time.time()-t0:.0f}s)", flush=True)
    return clean(s)


pw = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
if STAGE != "schuster":
    series = F.bin_daily_series(DDIR)
    A = F.feats_A(series, pw, 7)

if STAGE == "sanity":
    # train-only diagnostics (no test-period evaluation)
    print("meta (n_land, n_andes>=1500m) per bin:", series["meta"])
    for k in ("P_land", "P_andes", "SWE_andes"):
        s = series[k]
        print(k, "days", len(s), s.index[0].date(), s.index[-1].date(), "nan frac", float(np.isnan(s.to_numpy()).mean()),
              "annual mean (mm) bins 0,10,20,28:", (s.mean() * 365.25).iloc[[0, 10, 20, 28]].round(0).tolist())
    tr = A.t_start < "2013-01-01"
    print(A[tr].describe().T)
    print("NaN frac A:", A.isna().mean().to_dict())
    C = F.feats_C(DDIR, pw); print(C.dropna().head(3)); print(C.tail(3)); print("NaN C", C.isna().mean().to_dict())
    D = F.feats_D(series, pw, 7); print(D[D.t_start < "2013-01-01"].describe().T)
    sys.exit()

if STAGE == "primary":
    OUT["A_y_main_eval"] = ev(A, "y_main", pw, tag="A primary")
    OUT["A_y_main_sur_phase"] = sur(A, "y_main", pw, n=NS, tag="A primary")
    OUT["A_y_all_eval"] = ev(A, "y_all", pw, tag="A")
    OUT["A_y_all_sur_phase"] = sur(A, "y_all", pw, n=200, tag="A", seed=1)

if STAGE == "yall1000":
    OUT["A_y_all_sur_phase_n1000"] = sur(A, "y_all", pw, n=1000, tag="A", seed=11)
    OUT["A_y_all_sur_circ_n500"] = sur(A, "y_all", pw, n=500, mode="circular_shift", tag="A", seed=12)

if STAGE == "secondary":
    # other surrogate modes for A
    for tgt in ("y_main", "y_all"):
        OUT[f"A_{tgt}_sur_binperm"] = sur(A, tgt, pw, n=200, mode="bin_permute", tag="A", seed=2)
    P = series["P_land"].to_numpy(); days = series["P_land"].index

    def builder_yearshift(shift_days):
        if shift_days == 0:
            return A
        rng = np.random.default_rng(shift_days)
        k = int(rng.integers(1, 46)); off = int(rng.integers(-30, 31))
        s = int(round(k * 365.25)) + off
        return F.feats_A(series, pw, 7, P=np.roll(P, s, axis=0), days=days)
    for tgt in ("y_main", "y_all"):
        OUT[f"A_{tgt}_sur_yearshift"] = sur(None, tgt, pw, n=200, mode="circular_shift", builder=builder_yearshift,
                                            tag="A yearshift", seed=3)
    # differenced A (4-week change)
    Ad = F.difference_frame(A, 4)
    for tgt in ("y_main", "y_all"):
        OUT[f"Adiff_{tgt}_eval"] = ev(Ad, tgt, pw, tag="A diff")
        OUT[f"Adiff_{tgt}_sur_phase"] = sur(Ad, tgt, pw, n=200, tag="A diff", seed=4)
    # B seasonal, C ENSO/PDO, D snow, E Andes load
    sets = {"B": F.feats_B(pw, 7), "C": F.feats_C(DDIR, pw), "D": F.feats_D(series, pw, 7), "E": F.feats_E(series, pw, 7)}
    sets["Cdiff"] = F.difference_frame(sets["C"][["t_start", "x_oni", "x_pdo"]], 4)
    for name, fs in sets.items():
        for tgt in ("y_main", "y_all"):
            OUT[f"{name}_{tgt}_eval"] = ev(fs, tgt, pw, tag=name)
            OUT[f"{name}_{tgt}_sur_phase"] = sur(fs, tgt, pw, n=200, tag=name, seed=5)

if STAGE == "robust":
    for tgt in ("y_main", "y_all"):
        OUT[f"A_{tgt}_gbm"] = ev(A, tgt, pw, model="gbm", tag="A gbm")
    p55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
    A55 = F.feats_A(series, p55, 7)
    for tgt in ("y_main", "y_all"):
        OUT[f"A55_{tgt}_eval"] = ev(A55, tgt, p55, tag="A M5.5")
        OUT[f"A55_{tgt}_sur_phase"] = sur(A55, tgt, p55, n=200, tag="A M5.5", seed=6)
    p30 = H.get_panel(step_days=30, horizon_days=30, target_mag=6.0)
    A30 = F.feats_A(series, p30, 30)
    for tgt in ("y_main", "y_all"):
        OUT[f"A30_{tgt}_eval"] = ev(A30, tgt, p30, tag="A 30d")
        OUT[f"A30_{tgt}_sur_phase"] = sur(A30, tgt, p30, n=200, tag="A 30d", seed=7)
    # alternative split: train t+h <= 2006-01-01, test t >= 2006-01-01 (climatology still from 1979-2012 train window
    # would leak 2006-2012 -> refit climatology on <2006 for this check)
    palt = pw.copy()
    ts = palt.t_start; h = pd.Timedelta(days=7)
    palt["split"] = np.where(ts + h <= pd.Timestamp("2006-01-01"), "train", np.where(ts >= pd.Timestamp("2006-01-01"), "test", "gap"))
    F.TRAIN_END = pd.Timestamp("2006-01-01")
    Aalt = F.feats_A(series, palt, 7)
    F.TRAIN_END = pd.Timestamp("2013-01-01")
    for tgt in ("y_main", "y_all"):
        OUT[f"Aalt2006_{tgt}_eval"] = ev(Aalt, tgt, palt, tag="A split2006")
        OUT[f"Aalt2006_{tgt}_sur_phase"] = sur(Aalt, tgt, palt, n=200, tag="A split2006", seed=8)

if STAGE == "schuster":
    for mm in (6.0, 5.5):
        ev_ = H.get_events(min_mag=mm, declustered=True, start="1976-01-01")
        rng = np.random.default_rng(0)
        for lab, lo, hi in (("train", "1976-01-01", "2013-01-01"), ("test", "2013-01-01", "2026-10-18"), ("all", "1976-01-01", "2026-10-18")):
            e = ev_[(ev_.time >= lo) & (ev_.time < hi)]
            ph = 2 * np.pi * (e.time.dt.dayofyear.to_numpy() - 1 + e.time.dt.hour.to_numpy() / 24) / 365.25
            t0, t1 = pd.Timestamp(lo).value, pd.Timestamp(hi).value
            bt = pd.to_datetime(rng.integers(t0, t1, 20000))
            bph = 2 * np.pi * (bt.dayofyear.to_numpy() - 1 + bt.hour.to_numpy() / 24) / 365.25
            r = H.schuster_vs_background(ph, bph, n_mc=5000)
            mean_dir = float(np.degrees(np.angle(np.mean(np.exp(1j * ph)))) % 360 / 360 * 365.25)
            r = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in r.items()}
            OUT[f"schuster_M{mm}_{lab}"] = r | {"n": int(len(e)), "mean_doy": mean_dir}
            print(f"Schuster M>={mm} {lab}: n={len(e)} {r} mean_doy={mean_dir:.0f}", flush=True)

json.dump(OUT, open(HERE / f"results_{STAGE}.json", "w"), indent=1, default=float)
print("saved", HERE / f"results_{STAGE}.json")
