"""Calibration of the harness: (a) baseline metrics, (b) false-positive rate of evaluate+surrogate_test on pure noise.
Usage: python -I run_calibration.py baselines OUT.json
       python -I run_calibration.py noise OUT.json START END [target] [n_surr]
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H


def strip(r):
    return {k: v for k, v in r.items() if k not in ("preds",)}


def baselines(out):
    res = {}
    for (st, h) in [(7, 7), (1, 1), (30, 30)]:
        for tm in (6.0, 5.5):
            p = H.get_panel(step_days=st, horizon_days=h, target_mag=tm)
            for tg in ("y_all", "y_main"):
                for m in (("logit", "gbm") if st == 7 else ("logit",)):
                    r = H.evaluate(None, tg, m, p, return_preds=True)
                    bb = H.block_bootstrap_ci(r["preds"], "p_base", "p_poisson")
                    key = f"s{st}_h{h}_m{tm}_{tg}_{m}"
                    res[key] = {"n_test_events": r["n_test_events"], "n_train_events": r["n_train_events"],
                                "ig_base_vs_poisson": r["ig_base_vs_poisson"], "ig_etas_vs_poisson": r["ig_etas_vs_poisson"],
                                "base_vs_poisson_year_bootstrap": bb,
                                "ttest_base_vs_poisson": r["ttest_base_vs_poisson"],
                                "base": r["base"], "etas": r["etas"], "poisson": r["poisson"]}
                    print(key, H.summarize(r), "IG_etas=%.3f" % r["ig_etas_vs_poisson"], bb, flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)


def noise_features(panel, k, seed):
    rng = np.random.default_rng(1000 + seed)
    ts = np.sort(panel.t_start.unique())
    T, B = len(ts), panel.bin_id.nunique()
    kind = ["rw_global"] * 7 + ["wn_global"] * 7 + ["wn_bin"] * 3 + ["rw_bin"] * 3
    kd = kind[k % 20]
    if kd == "rw_global":
        return kd, pd.DataFrame({"t_start": ts, "x_noise": np.cumsum(rng.normal(size=T))})
    if kd == "wn_global":
        return kd, pd.DataFrame({"t_start": ts, "x_noise": rng.normal(size=T)})
    G = rng.normal(size=(T, B))
    if kd == "rw_bin":
        G = np.cumsum(G, axis=0)
    df = panel[["bin_id", "t_start"]].copy()
    df["x_noise"] = G.ravel()
    return kd, df


def noise(out, a, b, target="y_all", n=200):
    p = H.get_panel()
    res = []
    for k in range(a, b):
        t = time.time()
        kd, f = noise_features(p, k, k)
        r = H.evaluate(f, target, "logit", p)
        s = H.surrogate_test(f, n=n, mode="circular_shift", target=target, model="logit", panel=p, seed=k)
        row = {"k": k, "kind": kd, "target": target, "ig": r["ig_per_event"], "T": r["ttest_full_vs_base"]["T"],
               "p_T": r["ttest_full_vs_base"]["p_one_sided"], "p_shift": s["p_value"], "null_mean": s["null_mean"],
               "null_sd": s["null_sd"]}
        if kd in ("wn_bin", "rw_bin"):
            row["p_binperm"] = H.surrogate_test(f, n=n, mode="bin_permute", target=target, panel=p, seed=k)["p_value"]
        res.append(row)
        print(row, round(time.time() - t, 1), flush=True)
        json.dump(res, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "baselines":
        baselines(sys.argv[2])
    else:
        noise(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), *(sys.argv[5:6] or ["y_all"]),
              **({"n": int(sys.argv[6])} if len(sys.argv) > 6 else {}))
