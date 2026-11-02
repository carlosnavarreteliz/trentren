"""NTPP as a stand-alone forecast vs ETAS / ETAS-logit baseline / Poisson on the test period (weekly, M>=6 and M>=5.5).
Raw probabilities and a 2-parameter logistic recalibration of log10 P_ntpp fitted on train (cross-fitted) cells.
Usage: python3 -I standalone.py FEATS.parquet"""
import sys, os, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import harness as H

FP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
feats = pd.read_parquet(FP)
out = {}
for M, col in ((6.0, "x_ntpp_logp60"), (5.5, "x_ntpp_logp55")):
    panel = H.get_panel(step_days=7, horizon_days=7, target_mag=M)
    m = panel[["bin_id", "t_start", "split", "y_all", "y_main", "p_poisson_all", "p_poisson_main"]].merge(
        feats[["bin_id", "t_start", col]], on=["bin_id", "t_start"], how="left")
    assert len(m) == len(panel) and m[col].notna().all()
    tr = (m.split == "train").to_numpy(); te = (m.split == "test").to_numpy()
    lp = m[col].to_numpy()
    for target in ("y_all", "y_main"):
        r = H.evaluate(None, target=target, model="logit", panel=panel, return_preds=True)
        pr = r["preds"].copy()
        assert np.all(pr.t_start.to_numpy() == m.t_start.to_numpy()[te]) and np.all(pr.bin_id.to_numpy() == m.bin_id.to_numpy()[te])
        y = m[target].to_numpy()
        p_raw = 10 ** lp
        # 2-parameter logistic recalibration on train cells: logit p = a + b*log10 P_ntpp
        A = np.column_stack([lp, np.ones(len(lp))])
        w = H.logit_newton(A[tr], y[tr], C=1e6) if False else None
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp[tr, None], y[tr])
        p_cal = lr.predict_proba(lp[:, None])[:, 1]
        pr["p_ntpp_raw"] = p_raw[te]; pr["p_ntpp_cal"] = p_cal[te]
        N = int(pr.y.sum())
        res = {"n_test_events": N, "recal_coef": [float(lr.coef_[0, 0]), float(lr.intercept_[0])],
               "train_mean_p_raw": float(p_raw[tr].mean()), "train_rate": float(y[tr].mean()),
               "test_mean_p_raw": float(p_raw[te].mean()), "test_rate": float(y[te].mean())}
        for name in ("p_ntpp_raw", "p_ntpp_cal", "p_etas", "p_base", "p_poisson"):
            res[name] = H._metrics(pr.y.to_numpy(), pr[name].to_numpy())
        for a in ("p_ntpp_raw", "p_ntpp_cal"):
            for b in ("p_etas", "p_base", "p_poisson"):
                bb = H.block_bootstrap_ci(pr, a=a, b=b)
                res[f"IG_{a}_vs_{b}"] = bb
        out[f"M{M}_{target}"] = res
        print(f"M{M} {target} N={N}  AUC ntpp_cal {res['p_ntpp_cal']['roc_auc']:.3f} etas {res['p_etas']['roc_auc']:.3f} "
              f"base {res['p_base']['roc_auc']:.3f} pois {res['p_poisson']['roc_auc']:.3f}")
        for a in ("p_ntpp_raw", "p_ntpp_cal"):
            print("   ", a, {b: (round(res[f'IG_{a}_vs_{b}']['ig_per_event'], 4), round(res[f'IG_{a}_vs_{b}']['ci_lo'], 3),
                             round(res[f'IG_{a}_vs_{b}']['ci_hi'], 3), round(res[f'IG_{a}_vs_{b}']['p_boot_le0'], 3))
                         for b in ("p_etas", "p_base", "p_poisson")}, flush=True)
        pr.to_parquet(os.path.join(HERE, f"preds_standalone_M{M}_{target}.parquet"), index=False)
json.dump(out, open(os.path.join(HERE, "standalone.json"), "w"), indent=1, default=float)
