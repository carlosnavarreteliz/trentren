"""Harness tests for static structural features.

Run: /opt/miniconda3/bin/python3 -I analyze.py <profiles.csv> <out_dir> <stage> [n]
stages: primary (y_main M6, spatial FT n), secondary (y_all M6 + M5.5 + shift + bin_permute),
        single (each feature alone), robust (30-day panel, gbm, alternate split 2005)
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/structural_static")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import harness as H
import common as C

prof_csv, out_dir, stage = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
out_dir.mkdir(parents=True, exist_ok=True)
prof = C.load_profiles(prof_csv)
lat = prof.lat.to_numpy()


def strip(r):
    return {k: v for k, v in r.items() if k != "preds"}


def run_eval(panel, cols, target, model="logit", null_kinds=(("ft", N),), seed=0, label=""):
    vals = prof[cols].to_numpy(float)
    bv = C.bin_profile(lat, vals)
    feats = C.static_frame(panel, bv, cols)
    r = H.evaluate(feats, target=target, model=model, panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    out = {"label": label, "cols": cols, "target": target, "model": model, "eval": strip(r), "bootstrap": bb}
    print(label, H.summarize(r), "| boot", {k: round(v, 3) for k, v in bb.items()}, flush=True)
    for kind, n in null_kinds:
        t0 = time.time()
        if kind == "bin_permute":
            s = H.surrogate_test(feats, n=n, mode="bin_permute", target=target, model=model, panel=panel, seed=seed)
            s = {k: v for k, v in s.items() if k != "null"}
        else:
            null = C.run_null(panel, lat, vals, cols, target, model, n, kind, seed)
            s = C.pval(r["ig_per_event"], null)
        out[f"null_{kind}"] = s
        print(f"   null {kind}: p={s['p_value']:.4f} mean={s['null_mean']:.4f} q95={s['null_q95']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return out


res = {}
if stage == "primary":
    P = H.get_panel(7, 7, 6.0)
    res["primary_ymain_m6"] = run_eval(P, C.PRIMARY, "y_main", null_kinds=(("ft", N),), seed=1, label="PRIMARY y_main M6")
elif stage == "secondary":
    P = H.get_panel(7, 7, 6.0)
    res["m6_yall"] = run_eval(P, C.PRIMARY, "y_all", null_kinds=(("ft", N), ("shift", 500), ("bin_permute", 200)),
                              seed=2, label="M6 y_all")
    res["m6_ymain_shift"] = run_eval(P, C.PRIMARY, "y_main", null_kinds=(("shift", 500), ("bin_permute", 200)),
                                     seed=3, label="M6 y_main (shift/permute)")
    P55 = H.get_panel(7, 7, 5.5)
    for t in ("y_main", "y_all"):
        res[f"m55_{t}"] = run_eval(P55, C.PRIMARY, t, null_kinds=(("ft", 500),), seed=4, label=f"M5.5 {t}")
elif stage == "single":
    P = H.get_panel(7, 7, 6.0)
    for c in C.PRIMARY + C.SECONDARY:
        for t in ("y_main", "y_all"):
            res[f"{c}_{t}"] = run_eval(P, [c], t, null_kinds=(("ft", 200),), seed=5, label=f"single {c} {t}")
    res["all9_ymain"] = run_eval(P, C.PRIMARY + C.SECONDARY, "y_main", null_kinds=(("ft", 200),), seed=6,
                                 label="all 9 y_main")
elif stage == "robust":
    P = H.get_panel(7, 7, 6.0)
    for t in ("y_main", "y_all"):
        res[f"gbm_{t}"] = run_eval(P, C.PRIMARY, t, model="gbm", null_kinds=(("ft", 100),), seed=7, label=f"GBM {t}")
    P30 = H.get_panel(30, 30, 6.0)
    for t in ("y_main", "y_all"):
        res[f"m30_{t}"] = run_eval(P30, C.PRIMARY, t, null_kinds=(("ft", 500),), seed=8, label=f"30d {t}")
    PA = C.resplit_panel(P, "2005-01-01", "2013-01-01")
    for t in ("y_main", "y_all"):
        res[f"split2005_{t}"] = run_eval(PA, C.PRIMARY, t, null_kinds=(("ft", 500),), seed=9,
                                         label=f"alt split 2005-2012 {t}")
json.dump(res, open(out_dir / f"{stage}.json", "w"), indent=1, default=float)
print("saved", out_dir / f"{stage}.json")
