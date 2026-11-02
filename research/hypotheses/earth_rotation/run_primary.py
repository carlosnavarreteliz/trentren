"""Primary preregistered tests. Usage: python -I run_primary.py <eopc04 file> <out.json> [n_surr]"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/earth_rotation")
import numpy as np, pandas as pd, harness as H, features as F

eop = F.load_eop(sys.argv[1]); out_path = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
lod = eop.lod * 1000.0  # ms
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)

def h1_builder(rng):
    vals = lod.to_numpy() if rng is None else H.ft_surrogate(lod.to_numpy(), rng)
    return F.lod_features(pd.Series(vals, index=lod.index), panel, cols=("lod_decel_lag5y",))

h1 = h1_builder(None)
pf = F.pole_fit(eop, panel, 7)
h2 = F.pole_tide_panel(pf, panel)

def strip(r):
    return {k: v for k, v in r.items() if k != "preds"}

res = {}
for name, feats, builder in [("H1_lod_decel_lag5y", h1, h1_builder), ("H2_pole_tide", h2, None)]:
    for target in ("y_main", "y_all"):
        t0 = time.time()
        r = H.evaluate(feats, target=target, model="logit", panel=panel, return_preds=True)
        bb = H.block_bootstrap_ci(r["preds"])
        if builder is not None:
            s = H.surrogate_test(builder=builder, n=N, mode="phase_randomize", target=target, panel=panel, seed=11, n_jobs=2)
        else:
            s = H.surrogate_test(feats, n=N, mode="phase_randomize", target=target, panel=panel, seed=11, n_jobs=2)
        print(name, target, H.summarize(r, s), "boot", bb, f"{time.time()-t0:.0f}s", flush=True)
        res[f"{name}|{target}"] = {"eval": strip(r), "bootstrap": bb, "surrogate": {k: v for k, v in s.items() if k != "null"},
                                   "null": s["null"]}
        json.dump(res, open(out_path, "w"), indent=1, default=float)
print("done")
