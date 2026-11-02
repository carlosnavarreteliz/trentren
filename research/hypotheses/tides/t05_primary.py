"""Primary harness test of x_tsens (tidal sensitivity) + secondary features, with tide-shift and harness surrogates.
Usage: python -I t05_primary.py <data_dir> <out_dir> <stage> [n]
stages: eval      -> evaluate primary + secondary features (weekly M6 and M5.5, y_main/y_all, logit/gbm)
        tshift    -> tide-shift surrogate for x_tsens (target given by argv[5], default y_main), n surrogates
        harness   -> harness surrogate_test phase_randomize (n) and bin_permute (200) for x_tsens
"""
import sys, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H
from tidelib import Tides
from tfeatures import FeatureBuilder

D, OUT, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
n_s = int(sys.argv[4]) if len(sys.argv) > 4 else 200
tgt_arg = sys.argv[5] if len(sys.argv) > 5 else "y_main"
OUT.mkdir(parents=True, exist_ok=True)
T = Tides(D / "derived" / "tidal_extrema.npz")
evp = pd.read_parquet(OUT / "event_phases.parquet")


def strip(r):
    return {k: v for k, v in r.items() if k != "preds"}


def run_eval(panel, feats, cols, target, model="logit"):
    r = H.evaluate(feats[["bin_id", "t_start"] + cols], target=target, model=model, panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    out = strip(r) | {"bootstrap": bb}
    # leave-one-test-year-out IG
    p = r["preds"]
    ll = lambda y, q: np.where(y == 1, np.log(np.clip(q, 1e-12, 1)), np.log(np.clip(1 - q, 1e-12, 1)))
    d = ll(p.y.to_numpy(), p.p_full.to_numpy()) - ll(p.y.to_numpy(), p.p_base.to_numpy())
    loo = {}
    for yr in sorted(p.year.unique()):
        m = p.year.to_numpy() != yr
        loo[int(yr)] = float(d[m].sum() / max(p.y.to_numpy()[m].sum(), 1))
    out["loyo_ig"] = loo
    return out


if stage == "eval":
    res = {}
    for (step, h, M) in [(7, 7, 6.0), (7, 7, 5.5), (30, 30, 6.0)]:
        panel = H.get_panel(step_days=step, horizon_days=h, target_mag=M)
        FB = FeatureBuilder(T, evp, panel)
        e = FB.event_set(4.7, bg_only=True)
        f3 = FB.build(e, window=1095)
        f1 = FB.build(e, window=365, prefix="x1_").rename(columns={"x1_tsens": "x_tsens365", "x1_tsens_lp": "x_tsens365_lp"})
        fa = FB.amplitude(h)
        feats = f3.merge(f1[["bin_id", "t_start", "x_tsens365", "x_tsens365_lp"]], on=["bin_id", "t_start"]).merge(
            fa, on=["bin_id", "t_start"])
        tr = panel.split == "train"
        print(f"panel s{step} h{h} M{M}: frac NaN x_tsens {feats.x_tsens.isna().mean():.3f}; "
              f"x_tsens mean/sd {feats.x_tsens.mean():.3f}/{feats.x_tsens.std():.3f}", flush=True)
        for target in ("y_main", "y_all"):
            for name, cols in [("x_tsens", ["x_tsens"]), ("x_tsens_lp", ["x_tsens_lp"]), ("x_tamp", ["x_tamp"]),
                               ("x_tsens365", ["x_tsens365"]), ("all3", ["x_tsens", "x_tsens_lp", "x_tamp"])]:
                for model in (("logit", "gbm") if (step, M) == (7, 6.0) and name in ("x_tsens", "x_tamp") else ("logit",)):
                    r = run_eval(panel, feats, cols, target, model)
                    key = f"s{step}h{h}_M{M}|{target}|{name}|{model}"
                    res[key] = r
                    print(f"{key:40s} IG={r['ig_per_event']:+.4f} CI=[{r['bootstrap'].get('ci_lo', np.nan):+.3f},"
                          f"{r['bootstrap'].get('ci_hi', np.nan):+.3f}] AUC base/full={r['base']['roc_auc']:.3f}/"
                          f"{r['full']['roc_auc']:.3f} ASS={r['base']['molchan_ass']:.3f}/{r['full']['molchan_ass']:.3f} "
                          f"coef={r.get('coef_extra', r.get('importance_extra'))} n={r['n_test_events']}", flush=True)
        if (step, h, M) == (7, 7, 6.0):
            feats.to_parquet(OUT / "features_weekly_m6.parquet", index=False)
    json.dump(res, open(OUT / "eval_results.json", "w"), indent=1, default=str)

elif stage == "tshift":
    panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
    FB = FeatureBuilder(T, evp, panel)
    e = FB.event_set(4.7, bg_only=True)
    obs = H.evaluate(FB.build(e)[["bin_id", "t_start", "x_tsens"]], target=tgt_arg, panel=panel)["ig_per_event"]
    rng = np.random.default_rng(2026)
    shifts = rng.uniform(60, 3650, n_s) * rng.choice([-1, 1], n_s)
    null = []
    t0 = time.time()
    for k, s in enumerate(shifts):
        f = FB.build(e, shift=float(s))
        null.append(H.evaluate(f[["bin_id", "t_start", "x_tsens"]], target=tgt_arg, panel=panel)["ig_per_event"])
        if k % 100 == 0:
            print(k, round(time.time() - t0), flush=True)
    null = np.array(null)
    p = (1 + np.sum(null >= obs)) / (n_s + 1)
    r = {"ig_obs": obs, "p_value": float(p), "n": n_s, "null_mean": float(null.mean()), "null_sd": float(null.std()),
         "null_q95": float(np.quantile(null, 0.95)), "significant_bonferroni": bool(p < H.ALPHA_BONF),
         "null": null.round(5).tolist()}
    print({k: v for k, v in r.items() if k != "null"})
    json.dump(r, open(OUT / f"tshift_{tgt_arg}.json", "w"), indent=1)

elif stage == "harness":
    panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
    feats = pd.read_parquet(OUT / "features_weekly_m6.parquet")[["bin_id", "t_start", "x_tsens"]]
    res = {}
    for target in ("y_main", "y_all"):
        for mode, n in [("phase_randomize", n_s if target == tgt_arg else 200), ("bin_permute", 200)]:
            s = H.surrogate_test(feats, n=n, mode=mode, target=target, panel=panel, seed=11, n_jobs=1)
            res[f"{target}|{mode}"] = s
            print(target, mode, {k: v for k, v in s.items() if k != "null"}, flush=True)
    json.dump(res, open(OUT / "harness_surrogates.json", "w"), indent=1)
