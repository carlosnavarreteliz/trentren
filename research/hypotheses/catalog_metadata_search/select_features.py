"""Nested temporal-CV feature screening strictly inside the training period (no test data touched).
usage: python -I select_features.py FEATURES_PARQUET TARGET SPLIT(2013|2006) OUT_DIR
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/catalog_metadata_search")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import cmlib as C

feat_path, target, split, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
H_DAYS = 7
if split == "2006":
    H.TEST_START = pd.Timestamp("2006-01-01")
    panel = H.build_panel(7, 7, 6.0, use_cache=False)
    folds = C.FOLDS_2006
else:
    panel = H.get_panel(7, 7, 6.0)
    folds = C.FOLDS_2013
split_ts = pd.Timestamp("2006-01-01" if split == "2006" else "2013-01-01")
_tm = panel[panel.split == "test"].t_start.min(); assert split_ts <= _tm < split_ts + pd.Timedelta(days=7), _tm

F = pd.read_parquet(feat_path)
cols = [c for c in F.columns if c.startswith("x_")]
X = C.aligned_matrix(F, panel, cols)
train_all = (panel.t_start + pd.Timedelta(days=H_DAYS) <= split_ts).to_numpy()
nanfrac = np.mean(~np.isfinite(X[train_all]), axis=0)
elig = [c for c, f in zip(cols, nanfrac) if f <= 0.95]
print(f"{len(cols)} features, {len(elig)} eligible (<=95% NaN in train)", flush=True)

rows = []
t0 = time.time()
fold_years = {}
for fi, (vs, ve) in enumerate(folds):
    fp = C.fold_panel(panel, vs, ve, H_DAYS)
    msk = (panel.t_start < pd.Timestamp(ve)).to_numpy()
    Xf = X[msk]
    tr = (fp.split == "train").to_numpy()
    assert fp[fp.split == "test"].t_start.max() < split_ts
    base = H.evaluate(None, target, "logit", fp, return_preds=True)
    yrs = np.sort(base["preds"].year.unique())
    fold_years[fi] = yrs.tolist()
    print(f"fold {fi} {vs}..{ve}: n_val_events={base['n_test_events']} base IG vs Poisson={base['ig_base_vs_poisson']:+.3f}", flush=True)
    for c in elig:
        j = cols.index(c)
        x = Xf[:, [j]]
        if np.mean(~np.isfinite(x[tr])) > 0.95 or np.nanstd(x[tr]) == 0:
            for yv in yrs:
                yy = int(base["preds"][base["preds"].year == yv].y.sum())
                rows.append((c, fi, int(yv), 0.0, yy))
            continue
        x = C.winsorize(x, tr)
        r = H.evaluate(C.frame(fp, x, [c]), target, "logit", fp, return_preds=True)
        pr = r["preds"]
        d = H._ll(pr.y.to_numpy(), pr.p_full.to_numpy()) - H._ll(pr.y.to_numpy(), pr.p_base.to_numpy())
        g = pd.DataFrame({"year": pr.year, "d": d, "y": pr.y}).groupby("year").sum()
        for yv, rr in g.iterrows():
            rows.append((c, fi, int(yv), float(rr.d), int(rr.y)))
    print(f"  fold {fi} done, {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rows, columns=["feature", "fold", "year", "dll", "y"])
R.to_csv(f"{out_dir}/cv_rows_{target}_{split}.csv", index=False)

summ = []
for c, g in R.groupby("feature"):
    fy = g.groupby("year")[["dll", "y"]].sum()
    pooled = fy.dll.sum() / max(fy.y.sum(), 1)
    per_fold = g.groupby("fold").dll.sum()
    p = C.year_boot_p(fy.dll.to_numpy(), fy.y.to_numpy())
    summ.append(dict(feature=c, ig_val=pooled, n_pos_folds=int((per_fold > 0).sum()),
                     dll_f0=per_fold.get(0, 0), dll_f1=per_fold.get(1, 0), dll_f2=per_fold.get(2, 0),
                     p_boot=p, artifact_prone=C.is_artifact_prone(c)))
S = pd.DataFrame(summ).sort_values("ig_val", ascending=False).reset_index(drop=True)
S["bh_q10"] = C.bh(S.p_boot.to_numpy(), 0.10)
S["passes"] = S.bh_q10 & (S.ig_val > 0) & (S.n_pos_folds >= 2)

# greedy de-correlation among passing features (Spearman on train rows)
chosen = []
cand = S[S.passes].feature.tolist()
if cand:
    Xtr = pd.DataFrame(X[train_all][:, [cols.index(c) for c in cand]], columns=cand)
    rho = Xtr.rank().corr()
    for c in cand:
        if all(abs(rho.loc[c, k]) <= 0.8 for k in chosen):
            chosen.append(c)
        if len(chosen) == 10:
            break
top5 = S.feature.head(5).tolist()
S.to_csv(f"{out_dir}/cv_summary_{target}_{split}.csv", index=False)
out = {"target": target, "split": split, "n_features": len(cols), "n_eligible": len(elig),
       "folds": folds, "n_bh_q10": int(S.bh_q10.sum()), "n_pass": int(S.passes.sum()),
       "selected": chosen, "top5_no_fdr": top5, "eligible": elig,
       "min_p_boot": float(S.p_boot.min()), "runtime_s": time.time() - t0}
json.dump(out, open(f"{out_dir}/selection_{target}_{split}.json", "w"), indent=1)
print(S.head(25).to_string())
print(json.dumps({k: v for k, v in out.items() if k != "eligible"}, indent=1))
