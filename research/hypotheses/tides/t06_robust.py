"""Robustness: (1) tide-shift null for the event-level phase test, (2) time-of-day artefact check,
(3) phase test by depth class, (4) alternate split (train <=2004, test 2005-2026) for x_tsens.
Usage: python -I t06_robust.py <data_dir> <out_dir>"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H
from tidelib import Tides
from tfeatures import FeatureBuilder

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
T = Tides(D / "derived" / "tidal_extrema.npz")
ev = pd.read_parquet(OUT / "event_phases.parquet")
res = {}
rng = np.random.default_rng(99)
lo, hi = H.to_days(pd.Timestamp("1973-01-01")), H.to_days(pd.Timestamp("2026-10-17"))


def stat(ph, bins, bg_bins_ph=None):
    # background mean resultant per bin from random times
    z = np.exp(1j * ph)
    return np.abs(np.sum(z - M0[bins])) ** 2 / len(ph), np.degrees(np.angle(np.mean(z - M0[bins])))


M0 = np.zeros(29, complex)
for b in range(29):
    M0[b] = np.exp(1j * T.phase(np.full(5000, b), rng.uniform(lo, hi, 5000))).mean()

# (1) tide-shift null for the phase test (one global shift for all events keeps their mutual timing)
shifts = rng.uniform(60, 3650, 1000) * rng.choice([-1, 1], 1000)
tr = ev.time < H.TEST_START
for name, m in [("main_M4.7", ev.mainshock), ("bg_M4.7", ev.bg)]:
    for per, pm in [("train", tr), ("test", ~tr), ("all", tr | ~tr)]:
        sub = ev[m & pm]
        b, t = sub.bin_id.to_numpy(), sub.t_days.to_numpy()
        obs, ang = stat(sub.ph_total.to_numpy(), b)
        null = np.array([stat(T.phase(b, T.wrap(t + s)), b)[0] for s in shifts])
        p = (1 + np.sum(null >= obs)) / (len(null) + 1)
        res[f"tshift_phase|{name}|{per}"] = dict(n=len(sub), stat=float(obs), mean_phase=float(ang), p=float(p))
        print("tide-shift phase test", name, per, res[f"tshift_phase|{name}|{per}"], flush=True)

# (2) time-of-day: Schuster on UTC-hour phase (24 h and 12 h) of the mainshocks; local solar time ~ UTC-4.7h
sub = ev[ev.mainshock]
frac_day = np.mod(sub.t_days.to_numpy(), 1.0)
for per_h in (24, 12):
    r = H.schuster_test(2 * np.pi * frac_day * 24 / per_h)
    res[f"time_of_day_{per_h}h"] = r
    print("time of day", per_h, r, flush=True)
# lunar-day check: phase relative to the M2 period only (lunar semidiurnal, 12.4206 h) is captured by ph_ocean;
# solar S2 artefact check: phase test of ph_total restricted to nights vs days (local 20-06 h)
loc_h = np.mod(frac_day * 24 + sub.longitude.to_numpy() / 15.0, 24)
for nm, m in [("night", (loc_h >= 20) | (loc_h < 6)), ("day", (loc_h >= 8) & (loc_h < 18))]:
    s = sub[m]
    obs, ang = stat(s.ph_total.to_numpy(), s.bin_id.to_numpy())
    r = H.schuster_vs_background(s.ph_total.to_numpy(), T.phase(rng.choice(s.bin_id.to_numpy(), 20000),
                                 rng.uniform(lo, hi, 20000)), n_mc=3000, seed=3)
    res[f"daynight|{nm}"] = r | {"mean_phase": float(ang)}
    print(nm, res[f"daynight|{nm}"], flush=True)

# (3) depth classes (interface ~ 15-50 km)
for nm, m in [("d<15", sub.depth < 15), ("d15-50", (sub.depth >= 15) & (sub.depth <= 50)), ("d50-70", sub.depth > 50)]:
    s = sub[m]
    r = H.schuster_vs_background(s.ph_total.to_numpy(), T.phase(rng.choice(s.bin_id.to_numpy(), 20000),
                                 rng.uniform(lo, hi, 20000)), n_mc=3000, seed=4)
    res[f"depth|{nm}"] = r | {"mean_phase": float(stat(s.ph_total.to_numpy(), s.bin_id.to_numpy())[1])}
    print(nm, res[f"depth|{nm}"], flush=True)

# (4) alternate split for x_tsens (train t+h<=2005-01-01, test >=2005-01-01)
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0).copy()
S = pd.Timestamp("2005-01-01")
panel["split"] = np.where(panel.t_start >= S, "test",
                          np.where(panel.t_start + pd.Timedelta(days=7) <= S, "train", "gap"))
FB = FeatureBuilder(T, ev, panel)
f = FB.build(FB.event_set(4.7, True))
for target in ("y_main", "y_all"):
    r = H.evaluate(f[["bin_id", "t_start", "x_tsens"]], target=target, panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    res[f"altsplit2005|{target}"] = dict(ig=r["ig_per_event"], ci=[bb["ci_lo"], bb["ci_hi"]], n=r["n_test_events"],
                                          coef=r["coef_extra"])
    print("alt split", target, res[f"altsplit2005|{target}"], flush=True)
json.dump(res, open(OUT / "robust.json", "w"), indent=1, default=float)
