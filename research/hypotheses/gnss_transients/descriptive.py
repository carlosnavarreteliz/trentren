"""Descriptive: stand-alone AUC of the GNSS feature on covered bin-weeks (no baseline), train vs test,
and mean feature in positive cells vs all. usage: python -I descriptive.py <station_anoms.parquet> <tmpdir>"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, runpy
from sklearn.metrics import roc_auc_score
g = runpy.run_path("/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/gnss_transients/evaluate_gnss.py", run_name="lib")
H = g["H"]
pr = g["restrict"](H.get_panel(7, 7, 6.0))
for nm, (lag, W, clean) in {"z30": (2, 30, False), "z90": (2, 90, False), "z30_clean": (2, 30, True)}.items():
    G, C = g["bin_grid"](lag, W, clean)
    df, mask = g["to_frame"](pr, G, "x")
    m = pr[["bin_id", "t_start", "split", "y_main", "y_all", "etas_p1.08_c0.02_a1.1_s1.0"]].merge(df, on=["bin_id", "t_start"])
    m["cov"] = mask.ravel()
    for split in ("train", "test"):
        s = m[(m.split == split) & m["cov"]]
        out = [f"{nm:10s} {split:5s} covered cells={len(s):6d}"]
        for y in ("y_main", "y_all"):
            if s[y].sum() > 0:
                out.append(f"{y}: n+={int(s[y].sum()):3d} AUC(x)={roc_auc_score(s[y], s.x):.3f} "
                           f"mean x | y=1: {s.x[s[y]==1].mean():+.2f} vs all {s.x.mean():+.2f}")
        print("  ".join(out))
