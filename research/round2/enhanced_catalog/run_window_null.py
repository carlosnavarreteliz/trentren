"""Robustness of the significance test to the short-window design: windowed phase-randomized surrogates
(randomize only the period where the catalog exists) for IPOC (primary) and Riedel, plus a noise calibration.
Usage: python -I run_window_null.py <ipoc.txt> <riedel.csv> <part: ipoc|riedel|calib> [n]"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research"); sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, harness as H, ec_load, ec_features as F
OUT = "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/enhanced_catalog/"
part = sys.argv[3]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
rows = []

def test(name, f, P, start, tgt, n, seed):
    t0 = time.time()
    r = H.evaluate(f, target=tgt, model="logit", panel=P)
    b = F.window_builder(f, P, start)
    s = H.surrogate_test(builder=b, n=n, mode="phase_randomize", target=tgt, panel=P, seed=seed)
    row = dict(analysis=name, target=tgt, ig=round(r["ig_per_event"], 4), p_window=s["p_value"], n=n,
               null_mean=round(s["null_mean"], 4), null_sd=round(s["null_sd"], 4), null_q95=round(s["null_q95"], 4),
               n_test=r["n_test_events"], secs=round(time.time() - t0))
    print(row, flush=True); rows.append(row)
    return row

if part == "ipoc":
    cat = ec_load.load_ipoc(sys.argv[1])
    P, ob = F.sub_panel(7, 7, 5.0, 22, 27)
    f = F.build_features(cat, P, ob, 3.0, 365, "2008-01-01")
    for tgt in ("y_main", "y_all"):
        # also record the harness-default null mean for comparison
        s0 = H.surrogate_test(f, n=100, mode="phase_randomize", target=tgt, panel=P, seed=5)
        print("default-null", tgt, {k: v for k, v in s0.items() if k != "null"}, flush=True)
        rows.append(dict(analysis="IPOC primary, harness-default null (n=100)", target=tgt, ig=round(s0["ig_obs"], 4),
                         p_default=s0["p_value"], null_mean=round(s0["null_mean"], 4), null_sd=round(s0["null_sd"], 4)))
        test("IPOC primary ML>=3, windowed null", f, P, "2008-01-01", tgt, N, 41)
elif part == "riedel":
    cat = ec_load.load_riedel(sys.argv[2]).dropna(subset=["mag"])
    for M in (5.0, 5.5):
        P, ob = F.sub_panel(7, 7, M, 3, 27, end="2021-12-24", split_date="2019-07-01")
        f = F.build_features(cat, P, ob, 2.5, 180, "2017-07-01")
        for tgt in ("y_main", "y_all"):
            if M == 5.0:
                s0 = H.surrogate_test(f, n=100, mode="phase_randomize", target=tgt, panel=P, seed=5)
                rows.append(dict(analysis=f"Riedel ML>=2.5 M>={M}, harness-default null (n=100)", target=tgt, ig=round(s0["ig_obs"], 4),
                                 p_default=s0["p_value"], null_mean=round(s0["null_mean"], 4), null_sd=round(s0["null_sd"], 4)))
                print(rows[-1], flush=True)
            test(f"Riedel ML>=2.5 M>={M}, windowed null", f, P, "2017-07-01", tgt, N if M == 5.0 else 200, 42)
        if M == 5.0:
            f45 = F.build_features(cat, P, ob, 4.5, 180, "2017-07-01", n_short=40, n_long=300, minpos_short=8, minpos_long=30, minev_long=100)
            for tgt in ("y_main", "y_all"):
                test("Riedel ML>=4.5 (contrast) M>=5, windowed null", f45, P, "2017-07-01", tgt, 200, 43)
elif part == "calib":
    # pure-noise per-bin features in the Riedel design: FPR of the harness-default vs windowed null
    P, ob = F.sub_panel(7, 7, 5.0, 3, 27, end="2021-12-24", split_date="2019-07-01")
    rng = np.random.default_rng(99)
    ts = np.sort(P.t_start.unique()); B = P.bin_id.nunique(); T = len(ts)
    valid = ts >= np.datetime64("2017-07-01")
    for k in range(int(sys.argv[5]) if len(sys.argv) > 5 else 10):
        X = np.full((T, B, 3), np.nan)
        e = rng.normal(size=(valid.sum(), B, 3))
        if k % 2:  # AR(0.8) in time
            for t in range(1, len(e)):
                e[t] = 0.8 * e[t - 1] + e[t]
        X[valid] = e
        f = H._grid_to_frame(P, X, ["x_n1", "x_n2", "x_n3"])
        for tgt in ("y_all",):
            s0 = H.surrogate_test(f, n=N, mode="phase_randomize", target=tgt, panel=P, seed=k)
            b = F.window_builder(f, P, "2017-07-01")
            s1 = H.surrogate_test(builder=b, n=N, mode="phase_randomize", target=tgt, panel=P, seed=k + 1000)
            row = dict(analysis=f"noise {k} ({'AR0.8' if k % 2 else 'white'})", target=tgt, ig=round(s0["ig_obs"], 4),
                       p_default=s0["p_value"], null_mean_default=round(s0["null_mean"], 4),
                       p_window=s1["p_value"], null_mean_window=round(s1["null_mean"], 4))
            print(row, flush=True); rows.append(row)
json.dump(rows, open(OUT + f"window_null_{part}.json", "w"), indent=1, default=str)
