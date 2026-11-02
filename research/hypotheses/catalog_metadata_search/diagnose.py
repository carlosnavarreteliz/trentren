"""Artifact / leakage diagnostics for candidate features (preregistered checks 1-5).
usage: python -I diagnose.py FEATURES_PARQUET OUT_DIR TARGET feat1 [feat2 ...]
"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/catalog_metadata_search")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import cmlib as C

feat_path, out_dir, target = sys.argv[1:4]
names = sys.argv[4:]
F = pd.read_parquet(feat_path)
panel = H.get_panel(7, 7, 6.0)
X = C.aligned_matrix(F, panel, names)
tr = (panel.split == "train").to_numpy()
Xw = C.winsorize(X, tr)
bcols = H.baseline_feature_cols(panel)
B = H._transform_base(panel, bcols)
out = {}
for j, c in enumerate(names):
    d = {"artifact_prone_a_priori": C.is_artifact_prone(c)}
    s = pd.Series(Xw[:, j])
    yr = panel.year.to_numpy()
    ym = s.groupby(yr).mean()
    d["yearly_mean"] = {int(k): (None if not np.isfinite(v) else round(float(v), 4)) for k, v in ym.items()}
    d["nan_frac_by_period"] = {p: float(np.mean(~np.isfinite(X[m, j]))) for p, m in
                               [("1976-1999", yr < 2000), ("2000-2012", (yr >= 2000) & (yr < 2013)), ("2013+", yr >= 2013)]}
    # level shift vs earlier period (in train-sd units)
    sd = np.nanstd(Xw[tr, j])
    d["mean_shift_sd"] = {p: float((np.nanmean(Xw[m2, j]) - np.nanmean(Xw[m1, j])) / sd) if sd > 0 else None for p, m1, m2 in
                          [("2000+ vs pre2000", yr < 2000, (yr >= 2000) & (yr < 2013)),
                           ("2013+ vs 2000-12", (yr >= 2000) & (yr < 2013), yr >= 2013)]}
    # correlation with baseline clustering features (train rows)
    ok = tr & np.isfinite(X[:, j])
    rho = pd.DataFrame(np.column_stack([X[ok, j], B[ok]]), columns=[c] + bcols).rank().corr()[c].drop(c)
    d["max_abs_spearman_with_baseline"] = float(rho.abs().max())
    d["most_correlated_baseline_col"] = str(rho.abs().idxmax())
    # causally detrended version: x - trailing 156-week mean (same bin), shift(1)
    G = pd.DataFrame(Xw[:, j].reshape(-1, 29))
    det = (G - G.rolling(156, min_periods=52).mean().shift(1)).to_numpy().ravel()
    ef = C.frame(panel, det[:, None], [c + "_det"])
    r = H.evaluate(ef, target, "logit", panel)
    s = H.surrogate_test(ef, n=200, mode="phase_randomize", target=target, panel=panel, seed=21, n_jobs=2)
    d["detrended"] = {"ig_per_event": r["ig_per_event"], "p_phase": s["p_value"]}
    # does it carry y_main too (and y_all)?
    ef0 = C.frame(panel, Xw[:, [j]], [c])
    for tg in ("y_all", "y_main"):
        rr = H.evaluate(ef0, tg, "logit", panel)
        d[f"single_{tg}"] = {"ig_per_event": rr["ig_per_event"], "coef": rr["coef_extra"][c],
                             "auc_base": rr["base"]["roc_auc"], "auc_full": rr["full"]["roc_auc"]}
    out[c] = d
    print(c, json.dumps({k: v for k, v in d.items() if k != "yearly_mean"}), flush=True)
json.dump(out, open(f"{out_dir}/diagnose_{target}.json", "w"), indent=1)
