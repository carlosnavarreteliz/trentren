"""Secondary / robustness analyses (exploratory). Usage: python -I run_secondary.py <part> <out.json>
parts: families | variants | tune
"""
import sys, json, time, itertools
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/quiescence_migration")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import qm_features as Q

part, out_path = sys.argv[1], sys.argv[2]
res = {}


def slim(r):
    keep = ["n_test_events", "ig_per_event", "ig_full_vs_etas", "ig_base_vs_poisson", "ig_full_vs_poisson",
            "ig_etas_vs_poisson", "coef_extra", "importance_extra"]
    d = {k: r[k] for k in keep if k in r}
    d["auc_base"], d["auc_full"] = r["base"]["roc_auc"], r["full"]["roc_auc"]
    return d


def sur(F, target, panel, n=200, mode="phase_randomize", seed=0, model="logit"):
    s = H.surrogate_test(F, n=n, mode=mode, target=target, panel=panel, seed=seed, n_jobs=2, model=model)
    return {k: v for k, v in s.items() if k != "null"}


def save():
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if part == "families":
    panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
    F = Q.build(panel)
    for fam, cols in Q.FAMILIES.items():
        f = F[["bin_id", "t_start"] + cols]
        for target in ("y_main", "y_all"):
            r = H.evaluate(f, target=target, panel=panel)
            s = sur(f, target, panel, n=200, seed=hash(fam) % 1000)
            res[f"{fam}|{target}"] = slim(r) | {"sur_phase": s}
            print(fam, target, round(r["ig_per_event"], 4), r.get("coef_extra"), "p=", s["p_value"], flush=True)
            save()

elif part == "variants":
    # (a) weekly M6, GBM joint
    panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
    F = Q.build(panel)
    for target in ("y_main", "y_all"):
        r = H.evaluate(F, target=target, model="gbm", panel=panel)
        res[f"gbm_w_m6|{target}"] = slim(r)
        print("gbm", target, round(r["ig_per_event"], 4), flush=True)
    # (b) weekly M5.5
    p55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
    F55 = Q.build(p55)
    for target in ("y_main", "y_all"):
        r = H.evaluate(F55, target=target, panel=p55)
        s = sur(F55, target, p55, n=200, seed=55)
        res[f"w_m55|{target}"] = slim(r) | {"sur_phase": s}
        print("M5.5", target, round(r["ig_per_event"], 4), "p=", s["p_value"], flush=True)
        save()
    # (c) 30-day panel M6 (burn-in rescaled to 5 yr)
    p30 = H.get_panel(step_days=30, horizon_days=30, target_mag=6.0)
    F30 = Q.build(p30, p={"burn_in_steps": int(5 * 365.25 / 30)})
    for target in ("y_main", "y_all"):
        r = H.evaluate(F30, target=target, panel=p30)
        s = sur(F30, target, p30, n=200, seed=30)
        res[f"m30_m6|{target}"] = slim(r) | {"sur_phase": s}
        print("30d", target, round(r["ig_per_event"], 4), "p=", s["p_value"], flush=True)
        save()
    # (d) M>=5.0 input events (complete throughout)
    F50 = Q.build(panel, min_mag=5.0)
    for target in ("y_main", "y_all"):
        r = H.evaluate(F50, target=target, panel=panel)
        s = sur(F50, target, panel, n=200, seed=50)
        res[f"w_m6_inputs_m50|{target}"] = slim(r) | {"sur_phase": s}
        print("inputs M5.0", target, round(r["ig_per_event"], 4), "p=", s["p_value"], flush=True)
        save()
    # (e) alternate split: train 1976-2005, test 2006-2012 (inside original training period; includes Maule 2010)
    pa = panel.copy()
    end_t = pa.t_start + pd.Timedelta(days=7)
    A0, A1 = pd.Timestamp("2006-01-01"), pd.Timestamp("2013-01-01")
    pa["split"] = np.where(end_t <= A0, "train", np.where((pa.t_start >= A0) & (end_t <= A1), "test", "gap"))
    for target in ("y_main", "y_all"):
        r = H.evaluate(F, target=target, panel=pa)
        s = sur(F, target, pa, n=200, seed=6)
        res[f"altsplit_2006|{target}"] = slim(r) | {"sur_phase": s}
        print("alt split", target, round(r["ig_per_event"], 4), "p=", s["p_value"], flush=True)
        save()

elif part == "tune":
    panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
    pin = panel.copy()
    end_t = pin.t_start + pd.Timedelta(days=7)
    V0 = pd.Timestamp("2000-01-01")
    pin["split"] = np.where(end_t <= V0, "train", np.where((pin.t_start >= V0) & (end_t <= H.TEST_START), "test", "gap"))
    grids = {
        "rtl": [dict(rtl_r0=a, rtl_t0=b) for a, b in itertools.product([50.0, 100.0], [0.5, 1.0])],
        "zq": [dict(zq_w=w) for w in [0.5, 1.0, 2.0]],
        "doughnut": [dict(dn_w=w) for w in [3.0, 5.0]],
        "migr": [dict(mg_w=w) for w in [180.0, 365.0]],
        "sync": [dict(sy_w=w) for w in [3.0, 5.0]],
        "shadow": [dict(sh_tau=t) for t in [5.0, 10.0, 20.0]],
    }
    best = {}
    cache = {}
    for fam, grid in grids.items():
        scores = []
        for g in grid:
            key = json.dumps(g, sort_keys=True)
            if key not in cache:
                cache[key] = Q.build(panel, p=g)
            f = cache[key][["bin_id", "t_start"] + Q.FAMILIES[fam]]
            sc = sum(H.evaluate(f, target=t, panel=pin)["ig_per_event"] for t in ("y_all", "y_main"))
            scores.append(sc)
            print("tune", fam, g, round(sc, 4), flush=True)
        best[fam] = grid[int(np.argmax(scores))]
        res[f"tune_scores|{fam}"] = {"grid": grid, "val_ig_sum": scores}
    params = {}
    for fam, g in best.items():
        params.update(g)
    res["selected_params"] = params
    print("selected", params, flush=True)
    Ft = Q.build(panel, p=params)
    for target in ("y_main", "y_all"):
        r = H.evaluate(Ft, target=target, panel=panel, return_preds=True)
        bb = H.block_bootstrap_ci(r["preds"])
        s = sur(Ft, target, panel, n=200, seed=77)
        if s["p_value"] < 0.05:
            s = sur(Ft, target, panel, n=1000, seed=78)
        res[f"tuned_test|{target}"] = slim(r) | {"bootstrap": bb, "sur_phase": s}
        print("tuned test", target, round(r["ig_per_event"], 4), "p=", s["p_value"], bb, flush=True)
        save()
save()
