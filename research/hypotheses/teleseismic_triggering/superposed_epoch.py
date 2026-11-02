"""EXPLORATORY (not preregistered): superposed-epoch test of immediate remote dynamic triggering in Chile.
For each strong remote shallow event (remote from every bin, depth<=100 km, max strain proxy over bins >= thr),
count Chilean events in (t, t+W] minus (t-W, t]. Null: circular shift of the remote-trigger times (keeps their
clustering). Whole period 1976-2026 (no fitting, so no train/test issue), plus the 2013+ test period alone.
Usage: python -I superposed_epoch.py <cat1> <cat2> [n_null]
"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering")
import numpy as np, pandas as pd, harness as H
from tele_features import load_global, TeleFeatures, T0, SPAN_DAYS

HERE = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/teleseismic_triggering"
c1, c2 = sys.argv[1:3]
NN = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
tf = TeleFeatures(load_global([c1, c2]))
g = tf.g
maxe = tf.loge.max(axis=1)
tt = tf.t_days
rng = np.random.default_rng(5)

chile_sets = {}
ev = H.get_events(min_mag=4.7)
ev["td"] = (ev.time - T0).dt.total_seconds() / 86400.0
chile_sets["M4.7_all"] = ev.td.to_numpy()
chile_sets["M4.7_bg"] = ev.td[ev.bg].to_numpy()
chile_sets["M5.5_all"] = ev.td[ev.mag >= 5.5].to_numpy()
cat = H.load_catalog(min_mag=4.0)
cat = cat[cat.time >= "1995-01-01"]
chile_sets["M4.0_since1995"] = ((cat.time - T0).dt.total_seconds() / 86400.0).to_numpy()
for k in chile_sets:
    chile_sets[k] = np.sort(chile_sets[k])


def stat(trig, ce, W):
    a = np.searchsorted(ce, trig + W, "right") - np.searchsorted(ce, trig, "right")
    b = np.searchsorted(ce, trig, "right") - np.searchsorted(ce, trig - W, "right")
    return float(a.sum() - b.sum()), int(a.sum()), int(b.sum())


out = {}
t_test = (pd.Timestamp("2013-01-01") - T0).days
t_start = (pd.Timestamp("1976-01-01") - T0).days
for thr in (0.5, 1.0):
    sel = tf.remote_all & (g.depth.to_numpy() <= 100) & (maxe >= thr)
    trig_all = tt[sel]
    for period, (lo, hi) in {"1976-2026": (t_start, SPAN_DAYS), "2013-2026": (t_test, SPAN_DAYS)}.items():
        trig = trig_all[(trig_all >= lo + 5) & (trig_all < hi - 5)]
        for cname, ce in chile_sets.items():
            for W in (1.0, 5.0):
                obs, na, nb = stat(trig, ce, W)
                null = np.empty(NN)
                for i in range(NN):   # shift remote trigger times circularly within [lo, hi)
                    s = rng.uniform(365, (hi - lo) - 365)
                    tr = lo + np.mod(trig - lo + s, hi - lo)
                    null[i] = stat(tr, ce, W)[0]
                p = (1 + np.sum(null >= obs)) / (NN + 1)
                key = f"thr{thr}_{period}_{cname}_W{int(W)}d"
                out[key] = dict(n_triggers=int(len(trig)), after=na, before=nb, diff=obs,
                                null_mean=float(null.mean()), null_sd=float(null.std()), p_one_sided=float(p))
                print(key, out[key], flush=True)
json.dump(out, open(f"{HERE}/results/superposed_epoch.json", "w"), indent=1)
