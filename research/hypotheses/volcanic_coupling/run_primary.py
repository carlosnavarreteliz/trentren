"""Primary + secondary surrogate tests for volcanic_coupling.
Usage: OMP_NUM_THREADS=1 /opt/miniconda3/bin/python3 -I run_primary.py <data_dir> <out_json>
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/volcanic_coupling")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import volc_features as V

data_dir, out = sys.argv[1], sys.argv[2]
# surrogate counts per target: (phase_randomize, bin_permute, raw circular shift). y_main is the preregistered primary.
NS = {"y_main": (1000, 300, 150), "y_all": (300, 200, 100)}

v, e, arc = V.load_sources(data_dir)
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
feats = V.build_features(panel, e, arc)
res = {}


def strip(r):
    return {k: v for k, v in r.items() if k not in ("preds", "null")}


for target in ("y_main", "y_all"):
    n_ph, n_pm, n_sh = NS[target]
    t0 = time.time()
    r = H.evaluate(feats, target=target, model="logit", panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    bb_etas = H.block_bootstrap_ci(r["preds"], a="p_full", b="p_etas")
    pr = r["preds"].copy()
    lyo = {}
    for yr in sorted(pr.year.unique()):
        q = pr[pr.year != yr]
        ll = lambda p: np.sum(q.y * np.log(np.clip(q[p], 1e-15, 1)) + (1 - q.y) * np.log(np.clip(1 - q[p], 1e-15, 1)))
        lyo[int(yr)] = float((ll("p_full") - ll("p_base")) / q.y.sum())
    s_ph = H.surrogate_test(feats, n=n_ph, mode="phase_randomize", target=target, panel=panel, seed=11, n_jobs=2)
    s_pm = H.surrogate_test(feats, n=n_pm, mode="bin_permute", target=target, panel=panel, seed=12, n_jobs=2)
    builder = lambda sd: V.build_features(panel, e, arc, shift_days=sd)
    s_sh = H.surrogate_test(builder=builder, n=n_sh, mode="circular_shift", target=target, panel=panel,
                            seed=13, n_jobs=2)
    res[target] = dict(eval=strip(r), bootstrap_vs_base=bb, bootstrap_vs_etas=bb_etas, leave_year_out=lyo,
                       sur_phase=strip(s_ph), sur_binperm=strip(s_pm), sur_rawshift=strip(s_sh))
    print(target, H.summarize(r, s_ph), f"[{time.time()-t0:.0f}s]", flush=True)
    print("  bin_permute p", s_pm["p_value"], " raw-shift p", s_sh["p_value"], " boot", bb, flush=True)
    print("  coef", r["coef_extra"], flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
json.dump(res, open(out, "w"), indent=1, default=float)
