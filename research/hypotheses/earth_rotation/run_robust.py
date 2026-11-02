"""Secondary / robustness / exploratory tests. Usage: python -I run_robust.py <eopc04 file> <out.json> [n_surr]"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/earth_rotation")
import numpy as np, pandas as pd, harness as H, features as F

eop = F.load_eop(sys.argv[1]); out_path = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
lod = eop.lod * 1000.0
res = {}

def save():
    json.dump(res, open(out_path, "w"), indent=1, default=float)

def strip(r):
    d = {k: v for k, v in r.items() if k not in ("preds",)}
    return {k: d[k] for k in ("ig_per_event", "ig_full_vs_etas", "ig_base_vs_poisson", "n_test_events", "base", "full",
                              "coef_extra", "importance_extra", "ttest_full_vs_base") if k in d}

def run(key, panel, feats, target, builder=None, mode="phase_randomize", model="logit", n=N, surr=True):
    t0 = time.time()
    r = H.evaluate(feats, target=target, model=model, panel=panel, return_preds=True)
    entry = {"eval": strip(r), "bootstrap": H.block_bootstrap_ci(r["preds"])}
    if surr:
        s = H.surrogate_test(feats if builder is None else None, builder=builder, n=n, mode=mode, target=target,
                             model=model, panel=panel, seed=23, n_jobs=2)
        entry["surrogate"] = {k: v for k, v in s.items() if k != "null"}
    else:
        s = None
    print(key, H.summarize(r, s), entry["bootstrap"], f"{time.time()-t0:.0f}s", flush=True)
    res[key] = entry; save()
    return r

def feats_for(panel, h, lodcols=("lod_decel_lag5y",)):
    h1 = F.lod_features(lod, panel, cols=lodcols)
    pf = F.pole_fit(eop, panel, h)
    h2 = F.pole_tide_panel(pf, panel)
    return h1, h2, pf

def lod_builder_factory(panel, cols=("lod_decel_lag5y",), mode="phase"):
    def b(arg):
        if mode == "phase":
            vals = lod.to_numpy() if arg is None else H.ft_surrogate(lod.to_numpy(), arg)
            s = pd.Series(vals, index=lod.index)
        else:
            s = H.circular_shift_series(lod, arg) if arg else lod
        return F.lod_features(s, panel, cols=cols)
    return b

# --- weekly M6: circular shift (end-matched) robustness, GBM, exploratory features ------------------
pw = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
h1, h2, pf = feats_for(pw, 7, lodcols=("lod_decel_lag5y", "lod_decel_lag0", "lod365_anom"))
h1p = h1[["t_start", "x_lod_decel_lag5y"]]
for tg in ("y_main", "y_all"):
    run(f"W6|H1|{tg}|circshift", pw, h1p, tg, builder=lod_builder_factory(pw, mode="shift"), mode="circular_shift")
    run(f"W6|H2|{tg}|circshift", pw, h2, tg, mode="circular_shift")
    run(f"W6|H1|{tg}|gbm", pw, h1p, tg, model="gbm", surr=False)
    run(f"W6|H2|{tg}|gbm", pw, h2, tg, model="gbm", surr=False)
    run(f"W6|EXP_lod_decel_lag0|{tg}", pw, h1[["t_start", "x_lod_decel_lag0"]], tg,
        builder=lod_builder_factory(pw, cols=("lod_decel_lag0",)))
    run(f"W6|EXP_lod365_anom|{tg}", pw, h1[["t_start", "x_lod365_anom"]], tg,
        builder=lod_builder_factory(pw, cols=("lod365_anom",)))
    run(f"W6|EXP_cw_amp|{tg}", pw, pf[["t_start", "cw_amp"]].rename(columns={"cw_amp": "x_cw_amp"}), tg)

# --- weekly M5.5 ------------------------------------------------------------------------------------
p55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
a1, a2, _ = feats_for(p55, 7)
for tg in ("y_main", "y_all"):
    run(f"W55|H1|{tg}", p55, a1, tg, builder=lod_builder_factory(p55))
    run(f"W55|H2|{tg}", p55, a2, tg)

# --- 30-day panel M6 --------------------------------------------------------------------------------
p30 = H.get_panel(step_days=30, horizon_days=30, target_mag=6.0)
b1, b2, _ = feats_for(p30, 30)
for tg in ("y_main", "y_all"):
    run(f"M30|H1|{tg}", p30, b1, tg, builder=lod_builder_factory(p30))
    run(f"M30|H2|{tg}", p30, b2, tg)

# --- alternate split: train <= 2005, test >= 2006 (weekly M6) ---------------------------------------
pa = pw.copy()
cut = pd.Timestamp("2006-01-01"); h = pd.Timedelta(days=7)
pa["split"] = np.where(pa.t_start + h <= cut, "train", np.where(pa.t_start >= cut, "test", "gap"))
for tg in ("y_main", "y_all"):
    run(f"ALT2006|H1|{tg}", pa, h1p, tg, builder=lod_builder_factory(pa))
    run(f"ALT2006|H2|{tg}", pa, h2, tg)

# --- event-level pole-tide phase (exploratory, retrospective actual pole tide) ----------------------
act = F.actual_pole_tide_daily(eop, lat=-30.0)
w = 2 * np.pi / F.P_CW
rate = np.gradient(act.to_numpy()) * 1.0  # per day
phase = pd.Series(np.arctan2(-rate / w, act.to_numpy() - act.rolling(2190, min_periods=1000, center=True).mean().to_numpy()),
                  index=act.index)
ev = H.get_events(min_mag=6.0, declustered=True)
ev = ev[(ev.time >= "1976-01-01")]
for lab, sel in [("train", ev.time < H.TEST_START), ("test", ev.time >= H.TEST_START)]:
    e = ev[sel]
    ph_e = phase.reindex(e.time.dt.floor("D")).dropna().to_numpy()
    bgp = phase[(phase.index >= e.time.min().floor("D")) & (phase.index <= e.time.max().floor("D"))].dropna().to_numpy()
    sv = H.schuster_vs_background(ph_e, bgp, n_mc=5000)
    res[f"EVENT_phase|{lab}"] = {"n": int(len(ph_e)), "schuster_vs_bg": sv,
                                 "mean_phase_deg": float(np.degrees(np.angle(np.mean(np.exp(1j * ph_e)))))}
    print("event phase", lab, len(ph_e), sv, flush=True)
save(); print("done")
