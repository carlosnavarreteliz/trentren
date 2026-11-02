"""Test (a): Schuster / phase test of event timing vs tidal Coulomb-stress phase, against the phase occupation of
random times (H.schuster_vs_background). Also saves event phases for the feature builder.
Usage: python -I t03_phase_test.py <data_dir> <out_dir>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H
from tidelib import Tides

D, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(exist_ok=True, parents=True)
T = Tides(D / "derived" / "tidal_extrema.npz")
ev = H.get_events(min_mag=4.7)
ev = ev[(ev.depth <= 70) & (ev.bin_id >= 0) & (ev.bin_id < 29)].reset_index(drop=True)
for comp in ("total", "solid", "ocean"):
    ev[f"ph_{comp}"] = T.phase(ev.bin_id.to_numpy(), ev.t_days.to_numpy(), comp)
ev[["time", "t_days", "latitude", "longitude", "depth", "mag", "bin_id", "bg", "mainshock",
    "ph_total", "ph_solid", "ph_ocean"]].to_parquet(OUT / "event_phases.parquet", index=False)

rng = np.random.default_rng(12345)
t_lo, t_hi = H.to_days(pd.Timestamp("1973-01-01")), H.to_days(pd.Timestamp("2026-10-17"))


def background(bins, comp, n=20000):
    bb = rng.choice(bins, size=n, replace=True)
    tt = rng.uniform(t_lo, t_hi, size=n)
    return T.phase(bb, tt, comp)


def summarize(sub, comp):
    ph = sub[f"ph_{comp}"].to_numpy()
    bgp = background(sub.bin_id.to_numpy(), comp)
    r = H.schuster_vs_background(ph, bgp, n_mc=5000, seed=1)
    z = np.exp(1j * ph).mean() - np.exp(1j * bgp).mean()
    r.update(mean_phase_deg=float(np.degrees(np.angle(z))), excess_amp=float(np.abs(z)),
             frac_pos=float(np.mean(np.cos(ph) > 0)), frac_pos_bg=float(np.mean(np.cos(bgp) > 0)),
             schuster_uniform_p=H.schuster_test(ph)["p_value"])
    return r


res = {}
tr = ev.time < H.TEST_START
sets = {
    "main_M4.7": ev.mainshock, "main_M5.5": ev.mainshock & (ev.mag >= 5.5),
    "bg_M4.7": ev.bg, "all_M4.7": pd.Series(True, index=ev.index), "main_M6.0": ev.mainshock & (ev.mag >= 6.0),
}
for name, m in sets.items():
    for per, pm in [("train", tr), ("test", ~tr), ("all", pd.Series(True, index=ev.index))]:
        for comp in ("total", "solid", "ocean"):
            sub = ev[m & pm]
            if len(sub) < 10:
                continue
            r = summarize(sub, comp)
            res[f"{name}|{per}|{comp}"] = r
            print(f"{name:10s} {per:5s} {comp:6s} n={r['n']:5d} p={r['p_value']:.4f} phase={r['mean_phase_deg']:7.1f} "
                  f"amp={r['excess_amp']:.3f} frac+={r['frac_pos']:.3f} (bg {r['frac_pos_bg']:.3f})", flush=True)
json.dump(res, open(OUT / "phase_tests.json", "w"), indent=1)
