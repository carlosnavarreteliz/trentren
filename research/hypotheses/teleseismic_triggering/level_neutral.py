"""Exploratory leakage/level check: how much of the IG is just a test-period rate (level) correction?
For each model, rescale test probabilities by a single factor so sum(p)=observed (oracle, uses test total -> diagnostic
only), then recompute IG(full vs base). Args: cat1 cat2"""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H
from scipy.optimize import brentq
from tele_features import load_global, TeleFeatures
tf = TeleFeatures(load_global(sys.argv[1:3]))
panel = H.get_panel(7, 7, 6.0)
def ll(y, p): return np.sum(y*np.log(np.clip(p,1e-12,1)) + (1-y)*np.log(np.clip(1-p,1e-12,1)))
def best_scale(y, p):  # MLE single multiplicative factor on the odds
    lo = np.log(p/(1-p))
    f = lambda a: np.sum(1/(1+np.exp(-(lo+a)))) - y.sum()
    a = brentq(f, -3, 3); return 1/(1+np.exp(-(lo+a)))
for hyp, feats in (("h1", tf.h1(panel)), ("h2", tf.h2(panel))):
    for t in ("y_all", "y_main"):
        r = H.evaluate(feats, target=t, panel=panel, return_preds=True); p = r["preds"]; y = p.y.to_numpy()
        N = y.sum()
        pb, pf = best_scale(y, p.p_base.to_numpy()), best_scale(y, p.p_full.to_numpy())
        print(f"{hyp} {t}: IG raw {r['ig_per_event']:+.4f} | sum p_base {p.p_base.sum():.1f} sum p_full {p.p_full.sum():.1f} obs {N} "
              f"| oracle level gain of base {(ll(y,pb)-ll(y,p.p_base))/N:+.4f} | IG after level-neutralising both {(ll(y,pf)-ll(y,pb))/N:+.4f}")
