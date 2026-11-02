"""Follow-up of the exploratory secondary hit (weekly M5.5, y_all, joint block: IG +0.020, p=0.025 at n=200).
Not preregistered as primary; reported as exploratory. Usage: python -I run_m55_followup.py <out.json>"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/quiescence_migration")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import qm_features as Q

out = sys.argv[1]
res = {}
p55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
F = Q.build(p55)
strip = lambda s: {k: v for k, v in s.items() if k != "null"}
save = lambda: json.dump(res, open(out, "w"), indent=1, default=float)

r = H.evaluate(F, target="y_all", panel=p55, return_preds=True)
p = r["preds"]; y = p.y.to_numpy()
ll = lambda q: y * np.log(q) + (1 - y) * np.log1p(-q)
p["dll"] = ll(p.p_full.to_numpy()) - ll(p.p_base.to_numpy())
res["joint"] = {"ig": r["ig_per_event"], "coef": r["coef_extra"], "bootstrap": H.block_bootstrap_ci(p),
                "dll_by_year": p.groupby("year").dll.sum().round(3).to_dict(),
                "dll_pos": float(p.dll[y == 1].sum()), "dll_neg": float(p.dll[y == 0].sum()),
                "auc_base": r["base"]["roc_auc"], "auc_full": r["full"]["roc_auc"], "n_test_events": r["n_test_events"]}
# leave-one-year-out IG
loyo = {}
for yr in sorted(p.year.unique()):
    q = p[p.year != yr]
    loyo[int(yr)] = float(q.dll.sum() / q.y.sum())
res["joint"]["loyo_ig"] = loyo
print(res["joint"], flush=True)
save()
res["joint_phase_n1000"] = strip(H.surrogate_test(F, n=1000, mode="phase_randomize", target="y_all", panel=p55, seed=551, n_jobs=2))
print("phase n1000", res["joint_phase_n1000"], flush=True); save()
res["joint_binperm_n500"] = strip(H.surrogate_test(F, n=500, mode="bin_permute", target="y_all", panel=p55, seed=552, n_jobs=2))
print("binperm", res["joint_binperm_n500"], flush=True); save()
res["joint_circ_n200"] = strip(H.surrogate_test(F, n=200, mode="circular_shift", target="y_all", panel=p55, seed=553, n_jobs=2))
print("circ", res["joint_circ_n200"], flush=True); save()
for fam, cols in Q.FAMILIES.items():
    f = F[["bin_id", "t_start"] + cols]
    rr = H.evaluate(f, target="y_all", panel=p55)
    s = strip(H.surrogate_test(f, n=200, mode="phase_randomize", target="y_all", panel=p55, seed=560, n_jobs=2))
    res[f"fam_{fam}"] = {"ig": rr["ig_per_event"], "coef": rr["coef_extra"], "sur_phase": s}
    print(fam, round(rr["ig_per_event"], 4), rr["coef_extra"], "p=", s["p_value"], flush=True); save()
# robustness: inputs M>=5.0; alt split 2006-2012
F50 = Q.build(p55, min_mag=5.0)
rr = H.evaluate(F50, target="y_all", panel=p55)
res["inputs_m50"] = {"ig": rr["ig_per_event"], "sur_phase": strip(H.surrogate_test(F50, n=200, target="y_all", panel=p55, seed=570, n_jobs=2))}
print("inputs M5.0", res["inputs_m50"], flush=True); save()
pa = p55.copy()
end_t = pa.t_start + pd.Timedelta(days=7)
A0, A1 = pd.Timestamp("2006-01-01"), pd.Timestamp("2013-01-01")
pa["split"] = np.where(end_t <= A0, "train", np.where((pa.t_start >= A0) & (end_t <= A1), "test", "gap"))
rr = H.evaluate(F, target="y_all", panel=pa)
res["altsplit_2006"] = {"ig": rr["ig_per_event"], "n_test_events": rr["n_test_events"],
                        "sur_phase": strip(H.surrogate_test(F, n=200, target="y_all", panel=pa, seed=571, n_jobs=2))}
print("alt split", res["altsplit_2006"], flush=True); save()
rr = H.evaluate(F, target="y_all", panel=p55, model="gbm")
res["gbm"] = {"ig": rr["ig_per_event"]}
print("gbm", res["gbm"], flush=True); save()
