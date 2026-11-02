"""Positive control (deliberately leaky weak signal) and 'clustering in disguise' control."""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
p = H.get_panel()
rng = np.random.default_rng(7)
out = {}
# 1) positive control: future-leaking weak signal (must be detected)
f = p[["bin_id", "t_start"]].copy()
f["x_leak"] = p["y_all"].to_numpy() * 1.0 + rng.normal(0, 1.0, len(p))
r = H.evaluate(f, "y_all", "logit", p); s = H.surrogate_test(f, n=100, panel=p, seed=1)
out["positive_control_leak"] = {"ig": r["ig_per_event"], "p_shift": s["p_value"], "T": r["ttest_full_vs_base"]["T"]}
# 2) clustering in disguise: recent nearby seismicity (already in the baseline) -> should add ~nothing
ev = H.get_events(4.7)
f2 = H.past_event_count(ev[ev.mag >= 5.0], p, window_days=14, nbr=1, name="x_recent_m5")
f2["x_recent_m5"] = np.log1p(f2["x_recent_m5"])
r2 = H.evaluate(f2, "y_all", "logit", p); s2 = H.surrogate_test(f2, n=100, panel=p, seed=2)
# vs Poisson only (what a naive study would report)
from sklearn.metrics import roc_auc_score
te = (p.split == "test").to_numpy()
out["clustering_in_disguise"] = {"ig_vs_etas_baseline": r2["ig_per_event"], "p_shift": s2["p_value"],
                                 "naive_auc_alone_test": float(roc_auc_score(p.y_all[te], f2.x_recent_m5[te]))}
print(json.dumps(out, indent=1))
json.dump(out, open("controls.json", "w"), indent=1)
