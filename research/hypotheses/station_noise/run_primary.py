"""Primary + secondary tests. Usage: python -I run_primary.py <site_z.parquet> <out.json> [n_primary] [n_secondary]"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/station_noise")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import features as F

zpath, outp = sys.argv[1], sys.argv[2]
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 200
site_z = pd.read_parquet(zpath)
out = {}

def clean(r):
    return {k: v for k, v in r.items() if k not in ("preds", "null")}

def run_panel(tag, panel, feats, cols, targets, n_sur, extra_modes=True, gbm=True):
    f = feats[["bin_id", "t_start"] + cols]
    cov = f.merge(panel[["bin_id", "t_start", "split"]], on=["bin_id", "t_start"])
    print(f"[{tag}] non-NaN fraction train/test:", {c: (round(cov[cov.split=='train'][c].notna().mean(), 3),
          round(cov[cov.split=='test'][c].notna().mean(), 3)) for c in cols}, flush=True)
    for tgt in targets:
        t0 = time.time()
        r = H.evaluate(f, target=tgt, model="logit", panel=panel, return_preds=True)
        bb = H.block_bootstrap_ci(r["preds"])
        bbe = H.block_bootstrap_ci(r["preds"], b="p_etas")
        sur = H.surrogate_test(f, n=n_sur, mode="phase_randomize", target=tgt, panel=panel, seed=11, n_jobs=2)
        rec = {"logit": clean(r), "bootstrap_vs_base": bb, "bootstrap_vs_etas": bbe, "sur_phase": clean(sur)}
        print(f"[{tag}] {tgt}:", H.summarize(r, sur), "| boot", bb, flush=True)
        if extra_modes:
            s2 = H.surrogate_test(f, n=200, mode="bin_permute", target=tgt, panel=panel, seed=12, n_jobs=2)
            s3 = H.surrogate_test(f, n=200, mode="circular_shift", target=tgt, panel=panel, seed=13, n_jobs=2)
            rec["sur_bin_permute"] = clean(s2); rec["sur_circ"] = clean(s3)
            print(f"   bin_permute p={s2['p_value']:.3f}  circ p={s3['p_value']:.3f}", flush=True)
        if gbm:
            g = H.evaluate(f, target=tgt, model="gbm", panel=panel)
            rec["gbm"] = clean(g)
            print(f"   gbm IG={g['ig_per_event']:+.4f}", flush=True)
        out[f"{tag}|{tgt}"] = rec
        print(f"   ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(outp, "w"), indent=1, default=float)

# ---- PRIMARY: weekly M6, primary station set, x_hf + x_lp, y_main first ----
pw6 = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
fw6 = F.panel_features(site_z, pw6)
run_panel("W_M6_hf+lp", pw6, fw6, ["x_hf", "x_lp"], ["y_main", "y_all"], NP)
# ---- secondary: single features, negative control ----
for cols in (["x_hf"], ["x_lp"], ["x_ms"]):
    run_panel("W_M6_" + cols[0], pw6, fw6, cols, ["y_main", "y_all"], NS, extra_modes=False, gbm=False)
# ---- M5.5 weekly ----
pw55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
fw55 = F.panel_features(site_z, pw55)
run_panel("W_M5.5_hf+lp", pw55, fw55, ["x_hf", "x_lp"], ["y_main", "y_all"], NS, extra_modes=False)
# ---- dense station set (C1 added; test-only stations) ----
fd = F.panel_features(site_z, pw6, dense=True)
run_panel("W_M6_dense_hf+lp", pw6, fd, ["x_hf", "x_lp"], ["y_main", "y_all"], NS, extra_modes=False, gbm=False)
print("done")
