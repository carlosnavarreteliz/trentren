"""POST-HOC diagnostic (not preregistered): where does the x_or_anom gain on y_main come from?
Hypothesis to rule out: high outer-rise anomaly = shortly after a large interface event (outer-rise aftershocks),
when later M6 events are flagged as aftershocks by the non-causal declustering, so y_main is mechanically low.
Usage: python -I diag_outer_rise.py <out_dir>"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/vertical_coupling")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H, vc_features as V

out = sys.argv[1]
pw = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
F = pd.read_parquet(f"{out}/F1_weekly.parquet")
f = F[["bin_id", "t_start", "x_or_anom"]]
ev = H.get_events(min_mag=4.7)
big = ev[ev.mag >= 7.0]
# causal indicator: M>=7 within +-2 bins in the previous 2 years
td = H.to_days(pd.DatetimeIndex(np.sort(pw.t_start.unique())))
post = V.window_counts(big, td, 730, 0, nbr=2) > 0
P = V.to_frame(pd.DatetimeIndex(np.sort(pw.t_start.unique())), {"post_m7": post.astype(float)})

ll = lambda y, q: y * np.log(q) + (1 - y) * np.log(1 - q)
res = {}
for tgt in ("y_main", "y_all"):
    r = H.evaluate(f, target=tgt, panel=pw, return_preds=True)
    p = r["preds"].merge(P, on=["bin_id", "t_start"]).merge(f, on=["bin_id", "t_start"])
    p["d"] = ll(p.y, p.p_full) - ll(p.y, p.p_base)
    g = p.groupby("post_m7").agg(cells=("y", "size"), events=("y", "sum"), dLL=("d", "sum"),
                                 mean_x=("x_or_anom", "mean"))
    print(tgt, "coef", r["coef_extra"], "IG", round(r["ig_per_event"], 4)); print(g)
    res[tgt] = {"coef": r["coef_extra"], "ig": r["ig_per_event"], "by_post_m7": g.reset_index().to_dict(orient="list")}
    # top-quintile of x_or_anom: event rate vs base expectation, inside / outside post-M7 periods
    q = p.x_or_anom > p.x_or_anom.quantile(0.8)
    for pm in (0.0, 1.0):
        s = p[(p.post_m7 == pm)]
        for qq, nm in ((q, "top20"), (~q, "rest")):
            ss = s[qq.loc[s.index]]
            print(f"  post_m7={pm} {nm}: cells={len(ss)} obs={ss.y.sum()} exp_base={ss.p_base.sum():.2f} exp_full={ss.p_full.sum():.2f}")
# train-period association (allowed to look): rate of y_main by x_or_anom quintile, excluding post-M7 cells
tr = pw[pw.split == "train"].merge(f, on=["bin_id", "t_start"]).merge(P, on=["bin_id", "t_start"])
tr["q"] = pd.qcut(tr.x_or_anom.rank(method="first"), 5, labels=False)
print("TRAIN y_main rate by x_or_anom quintile (all / not post-M7):")
print(tr.groupby("q").y_main.mean().round(5).to_dict(), tr[tr.post_m7 == 0].groupby("q").y_main.mean().round(5).to_dict())
print("TRAIN y_main rate by post_m7:", tr.groupby("post_m7").y_main.mean().round(5).to_dict(),
      "y_all:", tr.groupby("post_m7").y_all.mean().round(5).to_dict())
print("corr(x_or_anom, post_m7) train:", round(float(np.corrcoef(tr.x_or_anom, tr.post_m7)[0, 1]), 3))
json.dump(res, open(f"{out}/diag_outer_rise.json", "w"), indent=1, default=float)
