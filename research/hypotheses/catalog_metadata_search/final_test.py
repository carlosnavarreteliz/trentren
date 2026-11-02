"""ONE-SHOT test-period evaluation of the preregistered feature sets (selection was frozen beforehand).
usage: python -I final_test.py FEATURES_PARQUET OUT_DIR TARGET [N_SURR]
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/catalog_metadata_search")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import cmlib as C

feat_path, out_dir, target = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
F = pd.read_parquet(feat_path)
sel = json.load(open(f"{out_dir}/selection_{target}_2013.json"))
elig = sel["eligible"]
t0 = time.time()
P = {"w6": H.get_panel(7, 7, 6.0), "w55": H.get_panel(7, 7, 5.5), "m6": H.get_panel(30, 30, 6.0)}
HD = {"w6": 7, "w55": 7, "m6": 30}
SPLIT = pd.Timestamp("2013-01-01")


def feats(cols, pk):
    pnl = P[pk]
    X = C.aligned_matrix(F, pnl, cols)
    tr = (pnl.t_start + pd.Timedelta(days=HD[pk]) <= SPLIT).to_numpy()
    return C.frame(pnl, C.winsorize(X, tr), cols)


def slim(r):
    keep = ["n_test_events", "ig_per_event", "ig_full_vs_etas", "ig_base_vs_poisson", "ig_full_vs_poisson",
            "ig_etas_vs_poisson", "coef_extra", "importance_extra", "ttest_full_vs_base"]
    o = {k: r[k] for k in keep if k in r}
    for m in ("base", "full"):
        o[m] = {k: r[m][k] for k in ("roc_auc", "molchan_ass", "brier", "expected", "observed")}
    return o


def nosur(s):
    return {k: v for k, v in s.items() if k != "null"}


def run_set(name, cols, n_surr, full=False):
    out = {"cols": cols}
    ef = feats(cols, "w6")
    r = H.evaluate(ef, target, "logit", P["w6"], return_preds=True)
    out["w6_logit"] = slim(r)
    out["w6_boot"] = H.block_bootstrap_ci(r["preds"])
    s = H.surrogate_test(ef, n=n_surr, mode="phase_randomize", target=target, panel=P["w6"], seed=11, n_jobs=2)
    if s["p_value"] < 0.02 and n_surr < 1000:
        s = H.surrogate_test(ef, n=1000, mode="phase_randomize", target=target, panel=P["w6"], seed=12, n_jobs=2)
    out["sur_phase"] = nosur(s)
    print(f"[{name}] {H.summarize(r, s)}  ({time.time()-t0:.0f}s)", flush=True)
    if full:
        out["sur_binperm"] = nosur(H.surrogate_test(ef, n=n_surr, mode="bin_permute", target=target, panel=P["w6"],
                                                    seed=13, n_jobs=2))
        out["sur_circ"] = nosur(H.surrogate_test(ef, n=n_surr, mode="circular_shift", target=target, panel=P["w6"],
                                                 seed=14, n_jobs=2))
        out["w6_gbm"] = slim(H.evaluate(ef, target, "gbm", P["w6"]))
        for pk in ("w55", "m6"):
            rr = H.evaluate(feats(cols, pk), target, "logit", P[pk], return_preds=True)
            out[f"{pk}_logit"] = slim(rr)
            out[f"{pk}_boot"] = H.block_bootstrap_ci(rr["preds"])
        pr = r["preds"]
        d = H._ll(pr.y.to_numpy(), pr.p_full.to_numpy()) - H._ll(pr.y.to_numpy(), pr.p_base.to_numpy())
        g = pd.DataFrame({"year": pr.year, "d": d, "y": pr.y}).groupby("year").sum()
        out["drop_year"] = {int(yv): float((g.d.sum() - g.d[yv]) / max(g.y.sum() - g.y[yv], 1)) for yv in g.index}
        out["per_year_dll"] = {int(yv): float(g.d[yv]) for yv in g.index}
        print(f"   binperm p={out['sur_binperm']['p_value']:.3f} circ p={out['sur_circ']['p_value']:.3f} "
              f"gbm IG={out['w6_gbm']['ig_per_event']:+.4f} w55 IG={out['w55_logit']['ig_per_event']:+.4f} "
              f"m6 IG={out['m6_logit']['ig_per_event']:+.4f}", flush=True)
    return out


res = {"target": target, "selection": {k: v for k, v in sel.items() if k != "eligible"}}
# baseline reference
rb = H.evaluate(None, target, "logit", P["w6"])
res["baseline_w6"] = {k: rb[k] for k in ("n_test_events", "ig_base_vs_poisson", "ig_etas_vs_poisson")} | \
    {"base_auc": rb["base"]["roc_auc"]}

# PRIMARY
if sel["selected"]:
    res["PRIMARY"] = run_set("PRIMARY selected", sel["selected"], N, full=True)
else:
    res["PRIMARY"] = {"cols": [], "note": "no feature passed BH-FDR q=0.10 + fold-consistency in training CV: "
                                          "null by construction (IG=0, no test evaluation needed)"}
    print("[PRIMARY] empty selected set -> null by construction", flush=True)

# E1 top-5 without FDR; E4 single best
res["E1_top5_noFDR"] = run_set("E1 top5 noFDR", sel["top5_no_fdr"], N, full=True)
res["E4_best_single"] = run_set("E4 best single", sel["top5_no_fdr"][:1], N, full=False)
# E3 families
phys = [c for c in elig if not C.is_artifact_prone(c)]
art = [c for c in elig if C.is_artifact_prone(c)]
meta = [c for c in elig if "_m_" in c]
res["E3_physical_all"] = run_set("E3 physical-candidate block", phys, 100)
res["E3_artifact_all"] = run_set("E3 artifact-prone block", art, 100)
res["E3_metadata_only"] = run_set("E3 ComCat-metadata block", meta, 100)
# E2 kitchen sink
res["E2_kitchen_logit"] = run_set("E2 kitchen sink logit", elig, 100)
efa = feats(elig, "w6")
res["E2_kitchen_gbm"] = slim(H.evaluate(efa, target, "gbm", P["w6"]))
print(f"[E2 kitchen gbm] IG={res['E2_kitchen_gbm']['ig_per_event']:+.4f}", flush=True)
json.dump(res, open(f"{out_dir}/final_{target}.json", "w"), indent=1, default=float)
print(f"done {time.time()-t0:.0f}s")
