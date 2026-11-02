"""Follow-up of the only nominally interesting secondary feature (x_v7, solar-wind speed, weekly M6).

usage: python -I run_v7_check.py <space_weather_daily.csv> <out_json> [n=200]
- coefficient sign, drop-one-test-year IG, end-matched circular-shift surrogate (raw daily series),
- causally detrended variant: 7-day mean V minus trailing 81-day mean V (ending 8 days before t_start),
- gbm check.
"""
import sys, json
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/space_weather")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import harness as H
import sw_lib as L

csv, out_json = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
daily = L.load_daily(csv)
panel = H.get_panel(7, 7, 6.0)
res = {}
f = L.make_feature(daily, "x_v7", panel)
for tg in ("y_all", "y_main"):
    r = H.evaluate(f, target=tg, model="logit", panel=panel, return_preds=True)
    P = r["preds"]
    ll = lambda y, p: y * np.log(p) + (1 - y) * np.log1p(-p)
    P["d"] = ll(P.y, P.p_full) - ll(P.y, P.p_base)
    drop = {int(yr): float(P[P.year != yr].d.sum() / P[P.year != yr].y.sum()) for yr in sorted(P.year.unique())}
    per_year = {int(yr): float(g.d.sum()) for yr, g in P.groupby("year")}
    rg = H.evaluate(f, target=tg, model="gbm", panel=panel)
    s_cs = H.surrogate_test(builder=L.builder_for(daily, "x_v7", panel, kind="shift"), n=n, mode="circular_shift",
                            target=tg, panel=panel, seed=41)
    res[tg] = {"ig": r["ig_per_event"], "coef": r["coef_extra"], "drop_one_year_ig": drop,
               "per_year_dll": per_year, "min_drop_one": min(drop.values()), "gbm_ig": rg["ig_per_event"],
               "p_circshift": s_cs["p_value"], "circ_null_q95": s_cs["null_q95"]}
    print(tg, "IG", round(r["ig_per_event"], 4), "coef", r["coef_extra"], "gbm", round(rg["ig_per_event"], 4),
          "p_circ", round(s_cs["p_value"], 4), "\n drop-one-year min", round(min(drop.values()), 4),
          "\n per-year dLL", {k: round(v, 2) for k, v in per_year.items()}, flush=True)

# causally detrended V
spec = ("v", None, dict(window_days=7, agg="mean", lag_days=1))


def build_det(vals):
    s = pd.Series(vals, index=daily.index)
    a = H.align_daily_series(s, panel, window_days=7, agg="mean", lag_days=1, name="a")
    b = H.align_daily_series(s, panel, window_days=81, agg="mean", lag_days=8, name="b")
    return pd.DataFrame({"t_start": a.t_start, "x_v7_det": a["a"] - b["b"]})


raw = daily["v"].to_numpy(float)
fd = build_det(raw)
for tg in ("y_all", "y_main"):
    r = H.evaluate(fd, target=tg, model="logit", panel=panel)
    s = H.surrogate_test(builder=lambda rng: build_det(L.aaft(raw, rng)), n=n, mode="phase_randomize", target=tg,
                         panel=panel, seed=43)
    res[f"det_{tg}"] = {"ig": r["ig_per_event"], "coef": r["coef_extra"], "p_ft": s["p_value"]}
    print("detrended", tg, res[f"det_{tg}"], flush=True)
json.dump(res, open(out_json, "w"), indent=1, default=float)
print("done")
