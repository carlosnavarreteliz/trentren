"""Leakage checks: (1) truncation test at test-period cut points with frozen params; (2) train-vs-test level shift of
components and composite; (3) ETAS expected vs observed counts by period.
Usage: python3 -I check_leakage.py <data_dir> <out.json>
"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/round2/surprise_etas")
import numpy as np, pandas as pd
import harness as H
import surprise_lib as S

dd, outp = sys.argv[1], sys.argv[2]
par = json.load(open(f"{dd}/etas_params_m4.7.json"))
F = pd.read_parquet(f"{dd}/features_m4.7_s7_h7.parquet")
panel = H.get_panel()
ev = H.get_events(min_mag=4.7)
res = {"truncation": {}}
cols = [c for c in F.columns if c.startswith("x_")]
for cut in ["2014-03-20", "2015-09-10", "2020-06-01"]:
    tcut = H.to_days(pd.Timestamp(cut)).item()
    evc = ev[ev.t_days < tcut].reset_index(drop=True)
    Fc, _, _ = S.build_features(evc, par, panel, verbose=False)
    Fc = Fc.merge(S.composite(Fc, panel), on=["bin_id", "t_start"])
    a = F[F.t_start <= pd.Timestamp(cut)].sort_values(["t_start", "bin_id"]).reset_index(drop=True)
    b = Fc[Fc.t_start <= pd.Timestamp(cut)].sort_values(["t_start", "bin_id"]).reset_index(drop=True)
    d = {}
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        nanmis = int(np.sum(np.isnan(x) != np.isnan(y)))
        ok = np.isfinite(x) & np.isfinite(y)
        d[c] = {"max_abs_diff": float(np.max(np.abs(x[ok] - y[ok]))) if ok.any() else 0.0, "nan_mismatch": nanmis}
    res["truncation"][cut] = d
    print(cut, max(v["max_abs_diff"] for v in d.values()), sum(v["nan_mismatch"] for v in d.values()), flush=True)

D = F.merge(panel[["bin_id", "t_start", "split"]], on=["bin_id", "t_start"])
lv = {}
for c in cols:
    tr, te = D.loc[D.split == "train", c], D.loc[D.split == "test", c]
    lv[c] = {"train_mean": float(tr.mean()), "test_mean": float(te.mean()), "train_sd": float(tr.std()),
             "shift_in_sd": float((te.mean() - tr.mean()) / tr.std())}
res["level_shift"] = lv
print(pd.DataFrame(lv).T.round(3).to_string())

# ETAS expected vs observed (whole region) per period
model = S.ETAS(ev, par)
edges = pd.to_datetime(["1976-01-01", "1990-01-01", "2000-01-01", "2013-01-01", "2020-01-01", "2026-10-17"])
C = model.compensator_bins(H.to_days(edges)).sum(1)
per = {}
for i in range(len(edges) - 1):
    n = int(((ev.time >= edges[i]) & (ev.time < edges[i + 1])).sum())
    per[f"{edges[i].date()}..{edges[i+1].date()}"] = {"observed": n, "etas_expected": float(C[i + 1] - C[i]),
                                                       "ratio": float(n / (C[i + 1] - C[i]))}
res["obs_vs_expected"] = per
print(json.dumps(per, indent=1))
json.dump(res, open(outp, "w"), indent=1)
