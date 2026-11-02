"""Fit harmonic tidal constituents (utide, OLS) on PRE-2013 hourly UHSLC data for 4 Chilean open-coast gauges and
predict the deterministic ocean tide on a 10-min UTC grid 1972-2028.
Usage: python -I t01_ocean_tides.py <data_dir>
Writes <data_dir>/derived/ocean_<name>.npy (float32, metres) and grid.npy (float64 days since 1970-01-01)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import utide

D = Path(sys.argv[1]); OUT = D / "derived"; OUT.mkdir(exist_ok=True)
GAUGES = {"arica": ("h083a.csv", -18.4667), "antofagasta": ("h080a.csv", -23.6533),
          "caldera": ("h088a.csv", -27.0667), "valparaiso": ("h081a.csv", -33.0267)}
EPOCH = pd.Timestamp("1970-01-01")
grid_t = pd.date_range("1972-01-01", "2028-01-01", freq="10min", inclusive="left")
grid_days = ((grid_t - EPOCH) / pd.Timedelta(days=1)).to_numpy(float)
np.save(OUT / "grid.npy", grid_days)
summary = {}
for name, (fn, lat) in GAUGES.items():
    d = pd.read_csv(D / fn, header=None, names=["y", "m", "d", "h", "v"])
    d = d[d.v != -32767]
    t = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d, hour=d.h))
    s = pd.Series(d.v.to_numpy(float) / 1000.0, index=t)
    s = s[(s.index >= "1980-01-01") & (s.index < "2013-01-01")]          # pre-2013 only
    s = s - s.groupby(s.index.to_period("M")).transform("mean")           # remove monthly mean (datum jumps, MSL)
    tt = ((s.index - EPOCH) / pd.Timedelta(days=1)).to_numpy(float)
    import time; t0 = time.time()
    coef = utide.solve(tt, s.to_numpy(), lat=lat, method="ols", conf_int="none", trend=False,
                       epoch="1970-01-01", verbose=False)
    rh = np.concatenate([utide.reconstruct(tt[k:k + 20000], coef, epoch="1970-01-01", verbose=False).h
                         for k in range(0, len(tt), 20000)])
    resid = s.to_numpy() - rh
    hrs = np.arange(grid_days[0] - 1, grid_days[-1] + 1.0 / 24 + 1, 1.0 / 24)       # hourly, then cubic to 10 min
    hp = np.concatenate([utide.reconstruct(hrs[k:k + 20000], coef, epoch="1970-01-01", verbose=False).h
                         for k in range(0, len(hrs), 20000)])
    from scipy.interpolate import CubicSpline
    pred = CubicSpline(hrs, hp)(grid_days).astype(np.float32)
    np.save(OUT / f"ocean_{name}.npy", pred)
    i = list(coef.name).index("M2")
    summary[name] = dict(n_hours=int(len(s)), span=[str(s.index.min()), str(s.index.max())],
                         var_explained=float(1 - np.var(resid) / np.var(s.to_numpy())),
                         M2_amp_m=float(coef.A[i]), M2_phase_deg=float(coef.g[i]),
                         pred_std_m=float(np.std(pred)))
    print(name, round(time.time() - t0), summary[name], flush=True)
json.dump(summary, open(OUT / "ocean_fit_summary.json", "w"), indent=1)
