"""Primary test (see PREREGISTRATION.md).
Usage: python3 -I run_primary.py <gcmt_chile.csv> <out.json> [n_phase_main] [n_other]"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/stress_rotation")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import features as F

path, out_path = sys.argv[1], sys.argv[2]
N_MAIN = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
N_OTH = int(sys.argv[4]) if len(sys.argv) > 4 else 200

cmt = F.load_cmt(path)
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
feats = F.build(cmt, panel)


def strip(r):
    return {k: v for k, v in r.items() if k not in ("preds", "null")}


def mech_shuffle_test(target, n, seed, panel=panel, build_kw={}, model="logit"):
    obs = H.evaluate(F.build(cmt, panel, **build_kw), target=target, model=model, panel=panel)["ig_per_event"]
    rng = np.random.default_rng(seed)
    null = []
    for k in range(n):
        f = F.build(F.shuffle_mechanisms(cmt, rng), panel, **build_kw)
        null.append(H.evaluate(f, target=target, model=model, panel=panel)["ig_per_event"])
    null = np.array(null)
    p = (1 + np.sum(null >= obs)) / (n + 1)
    return {"ig_obs": float(obs), "p_value": float(p), "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_q95": float(np.quantile(null, 0.95)), "n": n, "mode": "mechanism_shuffle"}


res = {}
for target in ("y_main", "y_all"):
    t0 = time.time()
    r = H.evaluate(feats, target=target, model="logit", panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    print(H.summarize(r)); print("  bootstrap:", bb, flush=True)
    d = {"eval": strip(r), "bootstrap": bb}
    nph = N_MAIN if target == "y_main" else N_OTH
    s = H.surrogate_test(feats, n=nph, mode="phase_randomize", target=target, panel=panel, seed=11, n_jobs=2)
    d["phase_randomize"] = strip(s); print("  phase_randomize:", strip(s), flush=True)
    if target == "y_all" and s["p_value"] < 0.02 and nph < 1000:
        s = H.surrogate_test(feats, n=1000, mode="phase_randomize", target=target, panel=panel, seed=12, n_jobs=2)
        d["phase_randomize_n1000"] = strip(s); print("  phase_randomize n1000:", strip(s), flush=True)
    s = H.surrogate_test(feats, n=N_OTH, mode="bin_permute", target=target, panel=panel, seed=13, n_jobs=2)
    d["bin_permute"] = strip(s); print("  bin_permute:", strip(s), flush=True)
    s = mech_shuffle_test(target, N_OTH, seed=14)
    d["mechanism_shuffle"] = s; print("  mechanism_shuffle:", s, flush=True)
    d["seconds"] = time.time() - t0
    res[target] = d
    json.dump(res, open(out_path, "w"), indent=1, default=float)
print("done")
