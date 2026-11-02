"""Secondary 3: Riedel-Hornig et al. (2026) margin-wide catalog, ML>=2.5, 2017-2021, custom split at 2019-07-01.
Usage: python -I run_riedel.py <Primary_Catalogue.csv> [n_surr]"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research"); sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H, ec_load, ec_features as F
OUT = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog/"
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
cat = ec_load.load_riedel(sys.argv[1])
print(cat.shape, cat.time.min(), cat.time.max(), cat.latitude.min(), cat.latitude.max())
cat["year"] = cat.time.dt.year
print(cat.groupby("year").agg(n=("mag", "size"), n25=("mag", lambda m: (m >= 2.5).sum()), n3=("mag", lambda m: (m >= 3).sum()),
                              mc=("mag", lambda m: H.mc_maxc(m.dropna().values))))
cat = cat.dropna(subset=["mag"])
rows = []
for M in (5.0, 5.5):
    P, ob = F.sub_panel(7, 7, M, 3, 27, end="2021-12-24", split_date="2019-07-01")
    for name, kw in [("Riedel ML>=2.5", dict(mthr=2.5)), ("Riedel ML>=4.5 (contrast)", dict(mthr=4.5, n_short=40, n_long=300, minpos_short=8, minpos_long=30, minev_long=100))]:
        mthr = kw.pop("mthr")
        f = F.build_features(cat, P, ob, mthr, 180, "2017-07-01", **kw)
        for tgt in ("y_main", "y_all"):
            r = H.evaluate(f, target=tgt, model="logit", panel=P, return_preds=True)
            bb = H.block_bootstrap_ci(r["preds"])
            n = 1000 if (r["ig_per_event"] > 0 and M == 5.0 and mthr == 2.5) else NS
            s = H.surrogate_test(f, n=n, mode="phase_randomize", target=tgt, panel=P, seed=31)
            row = dict(analysis=f"{name}, M>={M}", target=tgt, n_test=r["n_test_events"], n_train=r["n_train_events"],
                       ig=round(r["ig_per_event"], 4), ci_lo=round(bb["ci_lo"], 4), ci_hi=round(bb["ci_hi"], 4),
                       p_phase=s["p_value"], n_surr=n, auc_base=round(r["base"]["roc_auc"], 3), auc_full=round(r["full"]["roc_auc"], 3),
                       coef=r.get("coef_extra"))
            print(row, flush=True); rows.append(row)
json.dump(rows, open(OUT + "riedel.json", "w"), indent=1, default=str)
