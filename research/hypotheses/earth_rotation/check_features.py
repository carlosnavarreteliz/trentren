import sys, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/earth_rotation")
import numpy as np, pandas as pd, harness as H, features as F
eop = F.load_eop(sys.argv[1])
print(eop.describe())
panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
t0 = time.time(); pf = F.pole_fit(eop, panel, 7); print("pole fit s", time.time() - t0)
print(pf.describe())
# extrapolation accuracy: predicted osc vs actual pole tide at midpoint (exploratory, no targets)
act = F.actual_pole_tide_daily(eop, lat=-30.0)
pt = F.pole_tide_panel(pf, panel)
b = int(H.lat_to_bin(-30.0 + 0.01))
pb = pt[pt.bin_id == b].set_index("t_start")
mid = pb.index + pd.Timedelta(days=3)
a = act.reindex(mid.floor("D")).to_numpy()
ok = np.isfinite(a) & np.isfinite(pb.x_pt.to_numpy())
print("corr predicted vs actual pole tide (bin -30):", np.corrcoef(pb.x_pt.to_numpy()[ok], a[ok])[0, 1],
      "rms pred", np.std(pb.x_pt.to_numpy()[ok]), "rms err", np.std(pb.x_pt.to_numpy()[ok] - (a[ok] - np.nanmean(a[ok]))))
lf = F.lod_features(eop.lod * 1000, panel, cols=("lod_decel_lag5y", "lod_decel_lag0", "lod365_anom"))
print(lf.describe()); print(lf.isna().mean())
for c in lf.columns[1:]:
    v = lf[c].dropna().to_numpy(); print(c, "lag1 ac", np.corrcoef(v[1:], v[:-1])[0, 1])
