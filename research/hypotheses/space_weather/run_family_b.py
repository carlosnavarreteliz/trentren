"""Secondary (preregistered): Family B geomagnetic / solar features, each alone, weekly panel.

usage: python -I run_family_b.py <space_weather_daily.csv> <out_json> <target> <n_surr> [mag=6.0] [step=7] [feat,...]
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/space_weather")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import harness as H
import sw_lib as L

csv, out_json, target, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
mag = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
step = int(sys.argv[6]) if len(sys.argv) > 6 else 7
names = sys.argv[7].split(",") if len(sys.argv) > 7 else list(L.FAMILY_B)

daily = L.load_daily(csv)
panel = H.get_panel(step_days=step, horizon_days=step, target_mag=mag)
res = {}
t0 = time.time()
feats = []
for nm in names:
    spec = L.FAMILY_B[nm]
    if step != 7 and spec[2]["window_days"] == 7:     # match window to the step for alternate horizons
        spec = (spec[0], spec[1], dict(spec[2], window_days=step))
    f = L.make_feature(daily, nm, panel, spec=spec)
    feats.append(f.set_index("t_start"))
    r = H.evaluate(f, target=target, model="logit", panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    print(nm, H.summarize(r), "| boot", {k: round(v, 4) if isinstance(v, float) else v for k, v in bb.items()},
          flush=True)
    entry = {"evaluate": {k: v for k, v in r.items() if k != "preds"} | {"bootstrap": bb}}
    if n > 0:
        s = H.surrogate_test(builder=L.builder_for(daily, nm, panel, spec=spec), n=n, mode="phase_randomize",
                             target=target, panel=panel, seed=21)
        print("   surrogate p=%.4f obs=%+.4f null_mean=%+.4f q95=%+.4f" % (s["p_value"], s["ig_obs"],
              s["null_mean"], s["null_q95"]), f"({time.time()-t0:.0f}s)", flush=True)
        entry["surrogate"] = s
    res[nm] = entry
if len(names) > 1:  # joint (exploratory)
    J = feats[0].join(feats[1:]).reset_index()
    r = H.evaluate(J, target=target, model="logit", panel=panel, return_preds=True)
    print("JOINT", H.summarize(r), flush=True)
    res["joint"] = {"evaluate": {k: v for k, v in r.items() if k != "preds"} |
                    {"bootstrap": H.block_bootstrap_ci(r["preds"])}}
json.dump(res, open(out_json, "w"), indent=1, default=float)
print("done", time.time() - t0)
