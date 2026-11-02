"""Causality unit test: recompute NTPP states / residual / surprise features at random issue times from a catalog
truncated at t_start and compare with the stored feature table; also run the forecast simulator on the truncated
catalog with the same seed. Usage: python3 -I causality_check.py MODELDIR FEATS.parquet"""
import sys, os, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/surprise_neural")
import numpy as np
import pandas as pd
import torch
import harness as H
import ntpp_lib as L

MD, FP = sys.argv[1], sys.argv[2]
feats = pd.read_parquet(FP)
ev = H.get_events(min_mag=4.7)
Dfull = L.prep_events(ev)
rng = np.random.default_rng(7)
issues = np.sort(feats.t_start.unique())
picks = list(rng.choice(issues[issues >= np.datetime64("2013-01-01")], 6, replace=False)) + \
        list(rng.choice(issues[(issues >= np.datetime64("2003-01-01")) & (issues < np.datetime64("2013-01-01"))], 4,
                        replace=False)) + [np.datetime64("2014-03-31"), np.datetime64("2015-09-10")]
picks = [pd.Timestamp(p) for p in picks]
issues_ts = pd.DatetimeIndex(issues)
picks = [issues_ts[issues_ts.get_indexer([p], method="nearest")[0]] for p in picks]
out = []
for p in picks:
    k = "final" if p.year >= 2013 else "f3"
    T = float(H.to_days(p))
    evt = ev[ev.time < p]
    Dt = L.prep_events(evt)
    m = L.NTPP(json.load(open(os.path.join(MD, "selection.json")))["selected"]["H"])
    m.load_state_dict(torch.load(os.path.join(MD, f"ntpp_{k}_s0.pt")))
    st = L.score_sequence(m, Dt)
    sf = np.load(os.path.join(MD, f"score_{k}_s0.npz"))
    nt = Dt["n"]
    dH = float(np.abs(st["H"] - sf["H"][:nt]).max())
    dN = float(np.nanmax(np.abs((st["nll_t"] + st["nll_m"] + st["nll_l"])[1:nt] -
                                (sf["nll_t"] + sf["nll_m"] + sf["nll_l"])[1:nt])))
    # simulator on truncated vs full catalog (same seed, same single issue time)
    samp_t = L.LonDepSampler(Dfull, (Dfull["year"] <= 2012))
    P1, _ = L.simulate_forecasts(m, Dt, st["H"], np.array([T]), 7.0, (6.0,), samp_t, n_paths=200, seed=5)
    P2, _ = L.simulate_forecasts(m, Dfull, sf["H"], np.array([T]), 7.0, (6.0,), samp_t, n_paths=200, seed=5)
    dP = float(np.abs(P1 - P2).max())
    out.append({"t_start": str(p.date()), "model": k, "max_abs_dH": dH, "max_abs_dNLL": dN, "max_abs_dP_sim": dP})
    print(out[-1], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "causality_check.json"), "w"), indent=1)
