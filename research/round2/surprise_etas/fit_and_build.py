"""Fit ETAS on training data (m0 given) and build causal surprise features for the weekly panel (and optionally others).
Usage: python3 -I fit_and_build.py <m0> <out_dir> [step horizon target_mag]
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/surprise_etas")
import numpy as np, pandas as pd
import harness as H
import surprise_lib as S

m0 = float(sys.argv[1]); out = sys.argv[2]
step, hor, tm = (int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (7, 7, 6.0)
t0 = time.time()
ev = H.get_events(min_mag=4.7)
ev = ev[ev.mag >= m0].reset_index(drop=True)
pfile = f"{out}/etas_params_m{m0}.json"
try:
    par = json.load(open(pfile))
    print("loaded params", par)
except FileNotFoundError:
    par = S.fit_etas(ev, m0)
    json.dump(par, open(pfile, "w"), indent=1)
print(f"fit done {time.time()-t0:.0f}s", flush=True)
panel = H.get_panel(step_days=step, horizon_days=hor, target_mag=tm)
F, model, beta = S.build_features(ev, par, panel)
C = S.composite(F, panel)
F = F.merge(C, on=["bin_id", "t_start"])
F.to_parquet(f"{out}/features_m{m0}_s{step}_h{hor}.parquet")
print(f"features done {time.time()-t0:.0f}s; shape {F.shape}", flush=True)
print(F.describe().T.to_string())
