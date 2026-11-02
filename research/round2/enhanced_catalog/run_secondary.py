"""Secondary / robustness analyses (exploratory, no multiplicity credit). Usage: python -I run_secondary.py <ipoc.txt> [n_surr]"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research"); sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H, ec_load, ec_features as F
OUT = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog/"
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
cat = ec_load.load_ipoc(sys.argv[1])
rows = []

def run(name, feats, P, tgts=("y_main", "y_all"), model="logit", surr=True, cols=None):
    for tgt in tgts:
        r = H.evaluate(feats, target=tgt, model=model, panel=P, return_preds=True, extra_cols=cols)
        bb = H.block_bootstrap_ci(r["preds"])
        p = np.nan
        if surr and model == "logit":
            p = H.surrogate_test(feats, n=NS, mode="phase_randomize", target=tgt, panel=P, seed=21, extra_cols=cols)["p_value"]
        row = dict(analysis=name, target=tgt, model=model, n_test=r["n_test_events"], ig=round(r["ig_per_event"], 4),
                   ci_lo=round(bb["ci_lo"], 4), ci_hi=round(bb["ci_hi"], 4), p_phase=p,
                   auc_base=round(r["base"]["roc_auc"], 3), auc_full=round(r["full"]["roc_auc"], 3),
                   coef=r.get("coef_extra", r.get("importance_extra")))
        print(row, flush=True); rows.append(row)

P, ob = F.sub_panel(7, 7, 5.0, 22, 27)
base_cols = ["x_act7", "x_act30", "x_dbpos"]
f30 = F.build_features(cat, P, ob, 3.0, 365, "2008-01-01")
# 1) Mc contrast
f45 = F.build_features(cat, P, ob, 4.5, 365, "2008-01-01", n_short=40, n_long=300, minpos_short=8, minpos_long=30, minev_long=100)  # fewer events at ML>=4.5
f25 = F.build_features(cat, P, ob, 2.5, 365, "2008-01-01")
run("ML>=4.5 (high-Mc contrast)", f45, P)
run("ML>=2.5", f25, P)
# 2) single features
for c in base_cols:
    run(f"single {c} ML>=3", f30[["bin_id", "t_start", c]], P)
# 3) detrended
run("detrended (minus trailing 52 wk mean)", F.detrend(f30, base_cols), P)
# 4) GBM
run("GBM", f30, P, model="gbm", surr=False)
# 5) other targets
for M in (5.5, 6.0):
    PM, obM = F.sub_panel(7, 7, M, 22, 27)
    fM = F.build_features(cat, PM, obM, 3.0, 365, "2008-01-01")
    run(f"target M>={M}", fM, PM)
# 6) shallow-only events
fs = F.build_features(cat, P, ob, 3.0, 365, "2008-01-01", depth_max=60)
run("shallow events only (<=60 km)", fs, P)
# 7) daily panel, short windows
PD, obD = F.sub_panel(1, 1, 5.0, 22, 27)
fd = F.build_features(cat, PD, obD, 3.0, 365, "2008-01-01", windows=(1, 7), with_b=False)
run("daily panel act1+act7", fd, PD)
json.dump(rows, open(OUT + "secondary.json", "w"), indent=1, default=str)
# 8) leave-one-year-out from primary preds
for tgt in ("y_main", "y_all"):
    pr = pd.read_csv(OUT + f"preds_primary_{tgt}.csv")
    d = H._ll(pr.y.values, pr.p_full.values) - H._ll(pr.y.values, pr.p_base.values)
    g = pd.DataFrame({"year": pr.year, "d": d, "y": pr.y}).groupby("year").sum()
    loyo = {int(y): round(float((g.d.sum() - g.d[y]) / (g.y.sum() - g.y[y])), 4) for y in g.index}
    print(tgt, "per-year dLL", g.d.round(2).to_dict(), "LOYO IG", loyo)
    rows.append(dict(analysis="LOYO", target=tgt, loyo=loyo, per_year_dll=g.d.round(3).to_dict()))
json.dump(rows, open(OUT + "secondary.json", "w"), indent=1, default=str)
