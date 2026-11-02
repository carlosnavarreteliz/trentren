"""Cross-fitted NTPP: train fold models + final model, score the sequence, simulate weekly forecasts.
Usage: python3 -I run_ntpp.py OUTDIR [stage ...]   stages: train score forecast  (default: all)
Fold k model: trained on all pre-2013 blocks except block k; used for issue times / events in block k.
Final model: trained on 1973-2012; used for 2013+. 3 seeds per model (ensemble)."""
import sys, os, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/surprise_neural")
import numpy as np
import pandas as pd
import torch
import harness as H
import ntpp_lib as L

OUT = sys.argv[1]
stages = sys.argv[2:] or ["train", "score", "forecast"]
os.makedirs(OUT, exist_ok=True)
SEL = json.load(open(os.path.join(OUT, "selection.json")))["selected"]
HD, EP = SEL["H"], SEL["epochs"]
SEEDS = (0, 1, 2)
BLOCKS = {"f0": (1973, 1982), "f1": (1983, 1992), "f2": (1993, 2002), "f3": (2003, 2012), "final": (2013, 2100)}

ev = H.get_events(min_mag=4.7)
D = L.prep_events(ev)
yr = D["year"]


def train_mask(k):
    lo, hi = BLOCKS[k]
    return (yr <= 2012) & ~((yr >= lo) & (yr <= hi))


def path(k, s):
    return os.path.join(OUT, f"ntpp_{k}_s{s}.pt")


if "train" in stages:
    log = {}
    for k in BLOCKS:
        for s in SEEDS:
            t0 = time.time()
            m, hist, _ = L.train_model(D, train_mask(k), H=HD, epochs=EP, seed=s)
            torch.save(m.state_dict(), path(k, s))
            log[f"{k}_s{s}"] = {"train_nll_last": hist[-1]["train_nll"], "secs": time.time() - t0}
            print(k, s, log[f"{k}_s{s}"], flush=True)
    json.dump(log, open(os.path.join(OUT, "train_log.json"), "w"), indent=1)


def load(k, s):
    m = L.NTPP(HD)
    m.load_state_dict(torch.load(path(k, s)))
    m.eval()
    return m


if "score" in stages:
    for k in BLOCKS:
        for s in SEEDS:
            sc = L.score_sequence(load(k, s), D)
            np.savez_compressed(os.path.join(OUT, f"score_{k}_s{s}.npz"),
                                nll_t=sc["nll_t"], nll_m=sc["nll_m"], nll_l=sc["nll_l"], upd=sc["upd"],
                                latp=sc["latp"], pm55=sc["pm55"], pm60=sc["pm60"], tw=sc["tparams"]["tw"],
                                tmu=sc["tparams"]["tmu"], tsig=sc["tparams"]["tsig"], H=sc["H"])
            print("scored", k, s, flush=True)

if "forecast" in stages:
    panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
    issues = np.sort(panel.t_start.unique())
    idays = H.to_days(pd.DatetimeIndex(issues))
    iyear = pd.DatetimeIndex(issues).year.to_numpy()
    n_paths = int(os.environ.get("NTPP_PATHS", "100"))
    res = np.zeros((len(issues), 2, 29))
    trunc = {}
    for k, (lo, hi) in BLOCKS.items():
        sel = np.where((iyear >= lo) & (iyear <= hi))[0]
        if len(sel) == 0:
            continue
        acc = np.zeros((len(sel), 2, 29))
        for s in SEEDS:
            t0 = time.time()
            m = load(k, s)
            sc = np.load(os.path.join(OUT, f"score_{k}_s{s}.npz"))
            sampler = L.LonDepSampler(D, train_mask(k))
            P, nt = L.simulate_forecasts(m, D, sc["H"], idays[sel], 7.0, (5.5, 6.0), sampler,
                                         n_paths=n_paths, max_steps=300, batch_issues=16, seed=100 + s)
            acc += P / len(SEEDS)
            trunc[f"{k}_s{s}"] = nt
            print("forecast", k, s, len(sel), f"{time.time()-t0:.0f}s trunc={nt}", flush=True)
        res[sel] = acc
    np.savez_compressed(os.path.join(OUT, "forecast_weekly.npz"), P=res, issues=issues.astype("datetime64[ns]"))
    json.dump(trunc, open(os.path.join(OUT, "forecast_trunc.json"), "w"))
