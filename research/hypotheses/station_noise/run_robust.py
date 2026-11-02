"""Robustness checks. Usage: python -I run_robust.py <site_z.parquet> <out.json> [n_sur]"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research/hypotheses/station_noise")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import harness as H
import features as F

zpath, outp = sys.argv[1], sys.argv[2]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
site_z = pd.read_parquet(zpath)
out = {}
COLS = ["x_hf", "x_lp"]

def clean(r):
    return {k: v for k, v in r.items() if k not in ("preds", "null")}

def ll(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15); return y * np.log(p) + (1 - y) * np.log1p(-p)

def resplit(panel, year, h, start=None):
    p = panel.copy()
    ts = pd.Timestamp(f"{year}-01-01")
    p["split"] = np.where(p.t_start + pd.Timedelta(days=h) <= ts, "train", np.where(p.t_start >= ts, "test", "gap"))
    if start is not None:
        p = p[p.t_start >= pd.Timestamp(start)].reset_index(drop=True)
    return p

def one(tag, panel, f, tgts=("y_main", "y_all"), sur=True):
    for tgt in tgts:
        r = H.evaluate(f, target=tgt, model="logit", panel=panel, return_preds=True)
        rec = {"logit": clean(r), "bootstrap_vs_base": H.block_bootstrap_ci(r["preds"])}
        if sur:
            s = H.surrogate_test(f, n=NS, mode="phase_randomize", target=tgt, panel=panel, seed=21, n_jobs=2)
            rec["sur_phase"] = clean(s)
            print(f"[{tag}] {tgt}:", H.summarize(r, s), flush=True)
        else:
            print(f"[{tag}] {tgt}:", H.summarize(r), flush=True)
        out[f"{tag}|{tgt}"] = rec
        json.dump(out, open(outp, "w"), indent=1, default=float)
        yield r

pw6 = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)
fw6 = F.panel_features(site_z, pw6)[["bin_id", "t_start"] + COLS]

# 1) 30-day panel
p30 = H.get_panel(step_days=30, horizon_days=30, target_mag=6.0)
f30 = F.panel_features(site_z, p30, window_days=30)[["bin_id", "t_start"] + COLS]
list(one("M30_M6", p30, f30))
# 2) alternate split 2017 (weekly)
p17 = resplit(pw6, 2017, 7)
list(one("W_M6_split2017", p17, fw6))
# 3) rolling origin restricted to 1996+ issue times (station era), harness split 2013
p96 = resplit(pw6, 2013, 7, start="1996-01-01")
list(one("W_M6_from1996", p96, fw6))
# 4) lag 3 days
f_l3 = F.panel_features(site_z, pw6, lag_days=3)[["bin_id", "t_start"] + COLS]
list(one("W_M6_lag3", pw6, f_l3))
# 5) excluding cells within 30 d of an M>=6 in +-1 bin (aftershock-noise check), no refit: re-score the primary preds
for tgt in ("y_main", "y_all"):
    r = H.evaluate(fw6, target=tgt, model="logit", panel=pw6, return_preds=True)
    pr = r["preds"].merge(pw6[["bin_id", "t_start", "log_days_since_m6_nbr"]], on=["bin_id", "t_start"])
    quiet = pr.log_days_since_m6_nbr > np.log10(1 + 30)
    q = pr[quiet]
    d = ll(q.y.values, q.p_full.values) - ll(q.y.values, q.p_base.values)
    res = {"n_cells": int(quiet.sum()), "n_events": int(q.y.sum()), "ig_per_event": float(d.sum() / max(q.y.sum(), 1)),
           "bootstrap": H.block_bootstrap_ci(q)}
    out[f"W_M6_quiet30|{tgt}"] = res
    print(f"[quiet30] {tgt}:", res, flush=True)
    # per-year breakdown of the full-sample gain
    d_all = ll(pr.y.values, pr.p_full.values) - ll(pr.y.values, pr.p_base.values)
    yr = pd.DataFrame({"year": pr.year, "d": d_all, "y": pr.y}).groupby("year").sum()
    out[f"W_M6_peryear|{tgt}"] = yr.round(4).to_dict()
    print(yr.round(3).T.to_string(), flush=True)
json.dump(out, open(outp, "w"), indent=1, default=float)
print("done")
