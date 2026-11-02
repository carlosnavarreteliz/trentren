"""Hyperparameter / epoch selection for the NTPP: fit 1973-2007, validate on 2008-2012 events (no test data).
Usage: python3 -I select_ntpp.py OUTDIR"""
import sys, json, os, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/surprise_neural")
import numpy as np
import torch
import harness as H
import ntpp_lib as L

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
ev = H.get_events(min_mag=4.7)
ev = ev[ev.time < H.TEST_START]                      # selection never sees >= 2013
D = L.prep_events(ev)
yr = D["year"]
tr = yr <= 2007
val = (yr >= 2008) & (yr <= 2012)


def val_parts(model):
    s = L.score_sequence(model, D)
    return [float(np.nanmean(s[k][val])) for k in ("nll_t", "nll_m", "nll_l")]


def val_fn(model):
    return float(sum(val_parts(model)))


# trivial reference on val: stationary Poisson rate (train), GR exponential (train), empirical latitude hist
tr_ev = tr & (np.arange(D["n"]) > 0)
rate = (D["n"] * tr.mean()) / (D["t"][tr].max() - D["t"][tr].min())
taus = D["tau"][val]
ref_t = float(np.mean(-np.log(rate) + rate * taus))
beta = 1.0 / np.mean(D["m"][tr] - L.M0)
ref_m = float(np.mean(-np.log(beta) + beta * (D["m"][val] - L.M0)))
h, e = np.histogram((D["lat"][tr] - L.LAT_C) / L.LAT_S, bins=29, range=(-1, 1), density=True)
zi = np.clip((((D["lat"][val] - L.LAT_C) / L.LAT_S + 1) / 2 * 29).astype(int), 0, 28)
ref_l = float(np.mean(-np.log(np.maximum(h[zi], 1e-3))))
print("reference (Poisson/GR/emp-lat) val NLL parts", ref_t, ref_m, ref_l, "total", ref_t + ref_m + ref_l)

res = {"reference_val_parts": [ref_t, ref_m, ref_l]}
for Hd in (32, 64):
    for ep in (30, 60, 100):
        for seed in (0, 1):
            t0 = time.time()
            model, hist, best = L.train_model(D, tr, H=Hd, epochs=ep, val_fn=None, seed=seed)
            vp = val_parts(model)
            key = f"H{Hd}_e{ep}_s{seed}"
            res[key] = {"val_parts": vp, "val": sum(vp), "train_nll_last": hist[-1]["train_nll"],
                        "secs": time.time() - t0}
            print(key, [round(v, 4) for v in vp], round(sum(vp), 4), "train", round(hist[-1]["train_nll"], 4),
                  f"{time.time()-t0:.0f}s", flush=True)

cfg = {}
for Hd in (32, 64):
    for ep in (30, 60, 100):
        cfg[(Hd, ep)] = np.mean([res[f"H{Hd}_e{ep}_s{s}"]["val"] for s in (0, 1)])
bH, bE = min(cfg, key=cfg.get)
res["selected"] = {"H": int(bH), "epochs": int(bE), "mean_val": float(cfg[(bH, bE)])}
print("selected", res["selected"])
json.dump(res, open(os.path.join(out, "selection.json"), "w"), indent=1)
