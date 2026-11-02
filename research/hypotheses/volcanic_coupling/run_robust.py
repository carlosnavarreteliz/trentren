"""Robustness checks for volcanic_coupling.
Usage: /opt/miniconda3/bin/python3 -I run_robust.py <data_dir> <out_json> [n_sur=200]
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/volcanic_coupling")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import volc_features as V

data_dir, out = sys.argv[1], sys.argv[2]
n_s = int(sys.argv[3]) if len(sys.argv) > 3 else 200
v, e, arc = V.load_sources(data_dir)
ALL = ["x_arc_365", "x_arc_30", "x_erupt_365"]
res = {}


def run(name, panel, cols, target, model="logit", n=n_s):
    t0 = time.time()
    f = V.build_features(panel, e, arc)[["bin_id", "t_start"] + cols]
    r = H.evaluate(f, target=target, model=model, panel=panel, return_preds=True)
    nn = n if model == "logit" else max(50, n // 2)
    s = H.surrogate_test(f, n=nn, mode="phase_randomize", target=target, model=model, panel=panel, seed=21, n_jobs=2)
    bb = H.block_bootstrap_ci(r["preds"])
    row = dict(panel=name, target=target, model=model, cols=cols, ig=r["ig_per_event"],
               ig_full_vs_etas=r["ig_full_vs_etas"], ig_base_vs_etas=r["ig_full_vs_etas"] - r["ig_per_event"],
               p_phase=s["p_value"], n_sur=nn, null_mean=s["null_mean"], null_q95=s["null_q95"],
               auc_base=r["base"]["roc_auc"], auc_full=r["full"]["roc_auc"],
               ass_base=r["base"]["molchan_ass"], ass_full=r["full"]["molchan_ass"],
               n_test=r["n_test_events"], boot=bb, coef=r.get("coef_extra", r.get("importance_extra")))
    res.setdefault("rows", []).append(row)
    print(f"{name:10s} {target:6s} {model:5s} {','.join(c[2:] for c in cols):28s} IG={row['ig']:+.4f} "
          f"vsETAS={row['ig_full_vs_etas']:+.3f} p={row['p_phase']:.3f} AUC {row['auc_base']:.3f}->{row['auc_full']:.3f} "
          f"N={row['n_test']} [{time.time()-t0:.0f}s]", flush=True)


wk6 = H.get_panel(7, 7, 6.0)
for tgt in ("y_main", "y_all"):
    for c in ALL:                       # each feature alone
        run("wk_M6", wk6, [c], tgt)
    run("wk_M6", wk6, ALL, tgt, model="gbm")
wk55 = H.get_panel(7, 7, 5.5)
m30 = H.get_panel(30, 30, 6.0)
for tgt in ("y_main", "y_all"):
    run("wk_M5.5", wk55, ALL, tgt)
    run("30d_M6", m30, ALL, tgt)

# alternate split: train 1976..2004, pseudo-test 2005..2012 (never touches 2013+)
alt = wk6[wk6.t_start < "2013-01-01"].copy()
h = pd.Timedelta(days=7)
alt["split"] = np.where(alt.t_start + h <= pd.Timestamp("2005-01-01"), "train",
                        np.where(alt.t_start >= pd.Timestamp("2005-01-01"), "test", "gap"))
alt = alt.reset_index(drop=True)
for tgt in ("y_main", "y_all"):
    run("alt05_M6", alt, ALL, tgt)

# reverse direction (context only): eruption onsets within +-2 bins in 365 d after M>=7 events vs long-run rate
ev = H.get_events(min_mag=7.0)
ev = ev[(ev.time >= "1976-01-01") & (ev.time < "2025-10-01")]
eo = e[(e.onset >= "1976-01-01")]
span_yr = (pd.Timestamp("2026-10-17") - pd.Timestamp("1976-01-01")).days / 365.25
rev = []
for _, q in ev.iterrows():
    b = int(np.floor(q.latitude + 46))
    eb = np.floor(eo.lat + 46).astype(int)
    sel = (eb >= b - 2) & (eb <= b + 2)
    after = ((eo.onset > q.time) & (eo.onset <= q.time + pd.Timedelta(days=365)) & sel).sum()
    rate = sel.sum() / span_yr
    rev.append(dict(time=str(q.time.date()), lat=round(q.latitude, 2), mag=q.mag, eruptions_365d_after=int(after),
                    expected=round(rate, 2)))
res["reverse"] = rev
obs, exp_ = sum(r["eruptions_365d_after"] for r in rev), sum(r["expected"] for r in rev)
from scipy.stats import poisson
res["reverse_summary"] = dict(n_mainshocks=len(rev), observed=obs, expected=exp_,
                              p_poisson_ge=float(poisson.sf(obs - 1, exp_)))
print("reverse:", res["reverse_summary"])
for r in rev:
    print("  ", r)
json.dump(res, open(out, "w"), indent=1, default=float)
