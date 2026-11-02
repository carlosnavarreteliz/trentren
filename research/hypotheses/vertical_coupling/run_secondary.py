"""Secondary / robustness analyses (see PREREGISTRATION.md). Usage: python -I run_secondary.py <out_dir> [n_surr]"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/vertical_coupling")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import vc_features as V

out_dir = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
OUT = f"{out_dir}/secondary.json"
res = {}


def save():
    json.dump(res, open(OUT, "w"), indent=1, default=float)


def slim(r):
    keep = ["n_test_events", "n_train_events", "ig_per_event", "ig_full_vs_etas", "ig_base_vs_poisson",
            "ig_full_vs_poisson", "coef_extra", "importance_extra"]
    d = {k: r[k] for k in keep if k in r}
    d["auc_base"], d["auc_full"] = r["base"]["roc_auc"], r["full"]["roc_auc"]
    d["ass_base"], d["ass_full"] = r["base"]["molchan_ass"], r["full"]["molchan_ass"]
    d["T"] = r["ttest_full_vs_base"].get("T")
    return d


def run(name, feats, panel, targets=("y_main", "y_all"), model="logit", surr=True, mode="phase_randomize"):
    for tgt in targets:
        t0 = time.time()
        r = H.evaluate(feats, target=tgt, model=model, panel=panel, return_preds=True)
        d = slim(r)
        d["boot"] = H.block_bootstrap_ci(r["preds"])
        if surr:
            s = H.surrogate_test(feats, n=NS, mode=mode, target=tgt, model=model, panel=panel, seed=5, n_jobs=2)
            d["sur_p"], d["sur_null_mean"], d["sur_null_q95"], d["sur_mode"] = s["p_value"], s["null_mean"], s["null_q95"], mode
        res[f"{name}|{tgt}|{model}"] = d
        print(name, tgt, model, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != "boot"},
              "boot", {k: round(v, 3) for k, v in d["boot"].items()}, f"{time.time()-t0:.0f}s", flush=True)
        save()


ev = V.classify(H.get_events(min_mag=4.7))
pw = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
F1 = V.build_F1(pw, ev)

# A. single features (pure Bouchon test = x_ids_anom alone)
for c in ["x_ids_anom", "x_or_anom", "x_vert_shift"]:
    run(f"single_{c}", F1[["bin_id", "t_start", c]], pw)

# B. bg-only, 90-day short window
run("F1_bg_only", V.build_F1(pw, ev, bg_only=True), pw)
run("F1_short90", V.build_F1(pw, ev, short_ids=90), pw)

# I. causally differenced features (x(t) - x(t-26 weeks), same bin)
D = F1.sort_values(["bin_id", "t_start"]).copy()
for c in ["x_ids_anom", "x_or_anom", "x_vert_shift"]:
    D[c] = D.groupby("bin_id")[c].diff(26)
run("F1_diff26w", D, pw)

# C. GCMT outer-rise compression index
g = V.load_gcmt()
g = V.classify(g.assign(mag=g.mw))
orc = V.build_or_comp(pw, g)
run("gcmt_or_comp", orc, pw)

# F. GBM on the primary set (no surrogate: costly; IG only)
run("F1", F1, pw, model="gbm", surr=False)

# D. M5.5 weekly
p55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
run("F1_M5.5", V.build_F1(p55, ev), p55)

# E. 30-day panel M6
p30 = H.get_panel(step_days=30, horizon_days=30, target_mag=6.0)
run("F1_30d", V.build_F1(p30, ev), p30)

# G. alternate split: train t_start+h <= 2005-01-01, test t_start >= 2005-01-01
pa = pw.copy()
cut = pd.Timestamp("2005-01-01")
pa["split"] = np.where(pa.t_start >= cut, "test", np.where(pa.t_start + pd.Timedelta(days=7) <= cut, "train", "gap"))
run("F1_split2005", F1, pa)

# H. year-drop jackknife on primary preds
for tgt in ("y_main", "y_all"):
    r = H.evaluate(F1, target=tgt, panel=pw, return_preds=True)
    p = r["preds"]
    ll = lambda y, q: y * np.log(q) + (1 - y) * np.log(1 - q)
    p["d"] = ll(p.y, p.p_full) - ll(p.y, p.p_base)
    jk = {}
    for yr in sorted(p.year.unique()):
        q = p[p.year != yr]
        jk[int(yr)] = float(q.d.sum() / max(q.y.sum(), 1))
    per_year = p.groupby("year").agg(d=("d", "sum"), y=("y", "sum")).round(4)
    res[f"jackknife|{tgt}"] = {"drop_year_ig": jk, "per_year": per_year.to_dict(orient="index")}
    print("jackknife", tgt, {k: round(v, 3) for k, v in jk.items()}, flush=True)
    print(per_year.T, flush=True)
    save()
print("DONE", flush=True)
