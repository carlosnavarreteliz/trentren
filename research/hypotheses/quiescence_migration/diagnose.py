"""Post-hoc diagnostics (run AFTER the primary result was recorded): where does the joint block gain/lose,
standardized coefficients, and feature values in the weeks before the largest test-period mainshocks."""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/quiescence_migration")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import qm_features as Q

panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
F = pd.read_parquet("/Users/cnavarreteliz/Documents/Repos/earthquakes/research/data/quiescence_migration/features_default.parquet")
for target in ("y_main", "y_all"):
    r = H.evaluate(F, target=target, panel=panel, return_preds=True)
    print(target, "coef:", r["coef_extra"])
    p = r["preds"]
    y = p.y.to_numpy()
    ll = lambda q: y * np.log(q) + (1 - y) * np.log1p(-q)
    p["dll"] = ll(p.p_full.to_numpy()) - ll(p.p_base.to_numpy())
    print("  sum dLL on positive cells:", round(p.dll[y == 1].sum(), 2), " on negative cells:", round(p.dll[y == 0].sum(), 2))
    print("  dLL by year:", p.groupby("year").dll.sum().round(2).to_dict())
    print("  mean p_full/p_base on positives:", round((p.p_full / p.p_base)[y == 1].mean(), 3),
          " on negatives:", round((p.p_full / p.p_base)[y == 0].mean(), 3))
    tr = panel.split == "train"
    print("  expected/observed test: base", round(p.p_base.sum(), 1), "full", round(p.p_full.sum(), 1), "obs", int(y.sum()))

ev = H.get_events(min_mag=7.0)
ev = ev[(ev.time >= H.TEST_START) & ev.mainshock]
m = F.merge(panel[["bin_id", "t_start"]], on=["bin_id", "t_start"])
print("\nFeature values at the last issue time before each test-period M>=7 mainshock (bin of epicentre),"
      " with the train-period percentile of that value:")
trF = F[F.t_start < H.TEST_START]
for _, e in ev.iterrows():
    b = int(H.lat_to_bin(np.array([e.latitude]))[0])
    row = F[(F.bin_id == b) & (F.t_start <= e.time)].iloc[-1]
    pct = {c: round(float((trF[c].dropna() < row[c]).mean() * 100), 0) for c in Q.ALL_COLS}
    print(f"  {e.time.date()} M{e.mag} lat {e.latitude:.1f} bin {b}: ",
          {c: (round(float(row[c]), 2), pct[c]) for c in Q.ALL_COLS})
