"""Robustness checks (preregistered list): alternate horizon/panels, M>=5.5, GBM, alternate split, SOHO-era training.

usage: python -I run_robust.py <space_weather_daily.csv> <out_json> [n_surr=100]
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/space_weather")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import harness as H
import sw_lib as L

csv, out_json = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
daily = L.load_daily(csv)
res = {}
t0 = time.time()

NP1 = ("lognp", None, dict(window_days=1, agg="mean", lag_days=0))


def run(label, panel, name, spec, targets=("y_all", "y_main"), model="logit", nsur=n):
    # WORKAROUND harness bug: _BASE_CACHE is keyed by id(panel); a freed resplit copy's id can be reused by a new
    # panel of the same length -> stale baseline. Clear the cache whenever the panel changes.
    H._BASE_CACHE.clear()
    f = L.make_feature(daily, name, panel, spec=spec)
    for tg in targets:
        r = H.evaluate(f, target=tg, model=model, panel=panel, return_preds=True)
        bb = H.block_bootstrap_ci(r["preds"])
        e = {"ig": r["ig_per_event"], "n_test_events": r["n_test_events"], "auc_base": r["base"]["roc_auc"],
             "auc_full": r["full"]["roc_auc"], "ass_base": r["base"]["molchan_ass"],
             "ass_full": r["full"]["molchan_ass"], "ig_full_vs_etas": r["ig_full_vs_etas"],
             "boot": bb, "T": r["ttest_full_vs_base"]["T"]}
        if nsur > 0 and model == "logit":
            s = H.surrogate_test(builder=L.builder_for(daily, name, panel, spec=spec), n=nsur,
                                 mode="phase_randomize", target=tg, panel=panel, seed=31)
            e["p_sur"] = s["p_value"]; e["null_q95"] = s["null_q95"]
        res[f"{label}|{tg}|{model}"] = e
        print(f"{label:38s} {tg:6s} {model:5s} N={e['n_test_events']:4d} IG={e['ig']:+.4f} "
              f"CI=[{bb['ci_lo']:+.4f},{bb['ci_hi']:+.4f}] AUC {e['auc_base']:.3f}->{e['auc_full']:.3f} "
              f"p_sur={e.get('p_sur', float('nan')):.3f} ({time.time()-t0:.0f}s)", flush=True)


# 1) GBM on the primary (daily M6) -- no surrogates (slow); 2) daily M5.5 primary feature
pd1 = H.get_panel(1, 1, 6.0)
run("daily_M6_np_lag1", pd1, "x_np_lag1", NP1, model="gbm", nsur=0)
# 3) SOHO/ACE-era training (1996-2012) for the primary, test 2013+
run("daily_M6_np_lag1_train1996", L.resplit(pd1, "2013-01-01", "2013-01-01", train_start="1996-01-01"),
    "x_np_lag1", NP1, nsur=0)
del pd1
pd55 = H.get_panel(1, 1, 5.5)
run("daily_M5.5_np_lag1", pd55, "x_np_lag1", NP1, nsur=0)
del pd55
# 4) weekly panels: Np (7d) and Kp storms on M5.5
pw55 = H.get_panel(7, 7, 5.5)
for nm in ("x_np7", "x_kpmax7", "x_dstmin7", "x_storm7", "x_v7"):
    run(f"weekly_M5.5_{nm}", pw55, nm, L.FAMILY_B[nm])
# 5) 30-day panel M6: 30-day mean Np, max Kp, min Dst
p30 = H.get_panel(30, 30, 6.0)
for nm in ("x_np7", "x_kpmax7", "x_dstmin7", "x_v7"):
    c, tr, kw = L.FAMILY_B[nm]
    run(f"30d_M6_{nm.replace('7', '30')}", p30, nm.replace("7", "30"), (c, tr, dict(kw, window_days=30)))
# 6) alternate split: train 1976-2000, test 2001-2012 (weekly M6), real test period excluded
pw = H.get_panel(7, 7, 6.0)
alt = L.resplit(pw, "2001-01-01", "2001-01-01", test_end="2013-01-01")
for nm in ("x_np7", "x_kpmax7", "x_dstmin7", "x_storm7", "x_v7"):
    run(f"altsplit2001_weekly_M6_{nm}", alt, nm, L.FAMILY_B[nm])
# daily alternate split for the primary feature
pd1 = H.get_panel(1, 1, 6.0)
run("altsplit2001_daily_M6_np_lag1", L.resplit(pd1, "2001-01-01", "2001-01-01", test_end="2013-01-01"),
    "x_np_lag1", NP1, nsur=0)
json.dump(res, open(out_json, "w"), indent=1, default=float)
print("done", time.time() - t0)
