"""Secondary / robustness runs (see PREREGISTRATION.md).
Usage: python3 -I run_robust.py <gcmt_chile.csv> <out.json> [n_surr]"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/stress_rotation")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import features as F

path, out_path = sys.argv[1], sys.argv[2]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
cmt = F.load_cmt(path)
res = {}
try:
    res = json.load(open(out_path))
except Exception:
    pass


def strip(r):
    return {k: v for k, v in r.items() if k not in ("preds", "null")}


def run(name, panel, feats, targets=("y_main", "y_all"), model="logit", ns=NS, surr=True):
    if only and name not in only:
        return
    d = {}
    for tg in targets:
        r = H.evaluate(feats, target=tg, model=model, panel=panel, return_preds=True)
        e = {"eval": strip(r), "bootstrap": H.block_bootstrap_ci(r["preds"])}
        if surr:
            s = H.surrogate_test(feats, n=ns, mode="phase_randomize", target=tg, model=model, panel=panel,
                                 seed=21, n_jobs=2)
            e["phase_randomize"] = strip(s)
        d[tg] = e
        print(name, tg, H.summarize(r), "| boot", e["bootstrap"], "| phase p",
              e.get("phase_randomize", {}).get("p_value"), flush=True)
    res[name] = d
    json.dump(res, open(out_path, "w"), indent=1, default=float)


t0 = time.time()
pw = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
fw = F.build(cmt, pw)

# single-feature descriptive (no surrogates)
if not only or "single" in only:
    single = {}
    for c in F.COLS:
        for tg in ("y_main", "y_all"):
            r = H.evaluate(fw[["bin_id", "t_start", c]], target=tg, panel=pw)
            single[f"{c}|{tg}"] = {"ig": r["ig_per_event"], "coef": r["coef_extra"]}
    res["single_features_weekly_m6"] = single
    print("single", json.dumps(single, default=float), flush=True)

p55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
run("weekly_m55", p55, F.build(cmt, p55))
p30 = H.get_panel(step_days=30, horizon_days=30, target_mag=6.0)
run("30day_m6", p30, F.build(cmt, p30))
run("weekly_m6_W365", pw, F.build(cmt, pw, W=365))
run("weekly_m6_mw55", pw, F.build(cmt, pw, mmin=5.5))
run("weekly_m6_lag180", pw, F.build(cmt, pw, L=180))
run("weekly_m6_gbm", pw, fw, model="gbm", ns=100)

# alternate split: train 1976-2004 (purged), test 2005-2012; nothing from >= 2013 is used
pa = pw[pw.t_start < pd.Timestamp("2013-01-01") - pd.Timedelta(days=7)].copy()
end_tr = pd.Timestamp("2005-01-01")
pa["split"] = np.where(pa.t_start + pd.Timedelta(days=7) <= end_tr, "train",
                       np.where(pa.t_start >= end_tr, "test", "gap"))
pa = pa.reset_index(drop=True)
run("altsplit_2005_2012", pa, F.build(cmt, pa))

if not only or "misfit" in only:
    mf = F.build_misfit(cmt, pw)
    run("weekly_m6_plus_misfit_EXPLORATORY", pw, fw.merge(mf, on=["bin_id", "t_start"]))
    run("weekly_m6_misfit_only_EXPLORATORY", pw, mf)

pd1 = H.get_panel(step_days=1, horizon_days=1, target_mag=6.0)
run("daily_m6", pd1, F.build(cmt, pd1), targets=("y_main",))
print("done", time.time() - t0)
