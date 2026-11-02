"""Calibration v2: pure-noise features through evaluate + surrogate_test(mode='phase_randomize', n=200).
Usage: python -I run_calibration_v2.py OUT.json TARGET K_START K_END
Feature k (seeded by k): 0-19 global random walk, 20-39 global white noise, 40-49 global AR(1) phi=0.98,
50-54 per-bin random walk, 55-59 per-bin white noise. For k<20 and target y_main also circular_shift n=100."""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H


def feature(panel, k):
    rng = np.random.default_rng(50000 + k)
    ts = np.sort(panel.t_start.unique()); T = len(ts); B = panel.bin_id.nunique()
    if k < 20:
        return "rw_global", pd.DataFrame({"t_start": ts, "x_noise": np.cumsum(rng.normal(size=T))})
    if k < 40:
        return "wn_global", pd.DataFrame({"t_start": ts, "x_noise": rng.normal(size=T)})
    if k < 50:
        e = rng.normal(size=T); x = np.zeros(T); x[0] = e[0] / np.sqrt(1 - 0.98 ** 2)
        for i in range(1, T):
            x[i] = 0.98 * x[i - 1] + e[i]
        return "ar98_global", pd.DataFrame({"t_start": ts, "x_noise": x})
    G = rng.normal(size=(T, B))
    kind = "rw_bin" if k < 55 else "wn_bin"
    if kind == "rw_bin":
        G = np.cumsum(G, 0)
    df = panel[["bin_id", "t_start"]].copy(); df["x_noise"] = G.ravel()
    return kind, df


if __name__ == "__main__":
    out, target, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    p = H.get_panel()
    res = []
    for k in range(a, b):
        t = time.time()
        kind, f = feature(p, k)
        s = H.surrogate_test(f, n=200, mode="phase_randomize", target=target, panel=p, seed=k)
        row = {"k": k, "kind": kind, "target": target, "ig": s["ig_obs"], "p_pr": s["p_value"],
               "null_mean_pr": s["null_mean"], "null_sd_pr": s["null_sd"]}
        if kind.endswith("bin"):
            row["p_binperm"] = H.surrogate_test(f, n=100, mode="bin_permute", target=target, panel=p, seed=k)["p_value"]
        if k < 20 and target == "y_main":
            c = H.surrogate_test(f, n=100, mode="circular_shift", target=target, panel=p, seed=k)
            row["p_shift"] = c["p_value"]; row["null_mean_shift"] = c["null_mean"]
        res.append(row)
        print(row, round(time.time() - t, 1), flush=True)
        json.dump(res, open(out, "w"), indent=1, default=float)
