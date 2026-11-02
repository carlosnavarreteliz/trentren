"""ocean_loading analysis. Usage:
  python -I analysis.py <data_dir> <stage> [n_surr]
stages: check | primary | secondary | robust
Results appended as JSON to <script_dir>/results_<stage>.json
"""
import sys, os, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import harness as H

DATA = sys.argv[1]
STAGE = sys.argv[2]
NSUR = int(sys.argv[3]) if len(sys.argv) > 3 else 200
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("OL_SRC", "fd")


def load_daily(kind, src=SRC):
    d = pd.read_parquet(os.path.join(DATA, f"daily_{kind}_{src}.parquet"))
    d.columns = [int(c) for c in d.columns]
    return d


def win_mean(daily: pd.DataFrame, ts: pd.DatetimeIndex, window, lag):
    """mean over days D with t - lag - window <= D < t - lag, per bin (columns). Same semantics as H.align_daily_series."""
    v = daily.to_numpy(float)
    ok = np.isfinite(v)
    cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(np.where(ok, v, 0.0), 0)])
    cn = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(ok, 0)])
    d = daily.index.to_numpy()
    hi = np.searchsorted(d, (ts - pd.Timedelta(days=lag)).to_numpy(), side="left")
    lo = np.searchsorted(d, (ts - pd.Timedelta(days=lag + window)).to_numpy(), side="left")
    s = cs[hi] - cs[lo]
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)  # [T, B]


def make_feats(panel, daily: pd.DataFrame, prefix="x_", lag=1):
    ts = pd.DatetimeIndex(np.sort(panel.t_start.unique()))
    m7 = win_mean(daily, ts, 7, lag)
    m30b = win_mean(daily, ts, 30, lag + 7)
    m30 = win_mean(daily, ts, 30, lag)
    T, B = m7.shape
    df = pd.DataFrame({"t_start": np.repeat(ts.to_numpy(), B), "bin_id": np.tile(np.arange(B), T)})
    df[prefix + "dL7"] = (m7 - m30b).ravel()
    df[prefix + "L30"] = m30.ravel()
    return df


def mask_after_big(panel, feats, mmin=7.0, nbr=3, days=400):
    """Leakage check: NaN features for bin-times with an M>=mmin within +-nbr bins in [t-days, t). Causal."""
    ev = H.get_events(min_mag=mmin)
    f = feats.copy()
    cols = [c for c in f.columns if c.startswith("x_")]
    bad = np.zeros(len(f), bool)
    t = f.t_start.to_numpy()
    b = f.bin_id.to_numpy()
    for _, e in ev.iterrows():
        sel = (t > np.datetime64(e.time)) & (t <= np.datetime64(e.time) + np.timedelta64(days, "D")) & \
              (np.abs(b - int(e.bin_id)) <= nbr)
        bad |= sel
    f.loc[bad, cols] = np.nan
    return f, bad.mean()


def clean(d):
    return {k: v for k, v in d.items() if k not in ("null", "preds")}


def ev(feats, target, panel, model="logit"):
    r = H.evaluate(feats, target=target, model=model, panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    out = clean(r)
    out["bootstrap"] = bb
    out["auc_base"], out["auc_full"] = r["base"]["roc_auc"], r["full"]["roc_auc"]
    out["ass_base"], out["ass_full"] = r["base"]["molchan_ass"], r["full"]["molchan_ass"]
    # drop-one-year IG
    p = r["preds"]
    lf = p.y * np.log(p.p_full) + (1 - p.y) * np.log1p(-p.p_full)
    lb = p.y * np.log(p.p_base) + (1 - p.y) * np.log1p(-p.p_base)
    g = pd.DataFrame({"year": p.year, "d": lf - lb, "y": p.y}).groupby("year").sum()
    tot_d, tot_y = g.d.sum(), g.y.sum()
    out["drop_year_ig"] = {int(yr): float((tot_d - g.d[yr]) / (tot_y - g.y[yr])) for yr in g.index}
    out["per_year_dll"] = {int(yr): float(g.d[yr]) for yr in g.index}
    return out, r


def save(name, obj):
    fn = os.path.join(HERE, f"results_{name}.json")
    json.dump(obj, open(fn, "w"), indent=1, default=float)
    print("saved", fn)


if __name__ == "__main__":
    t0 = time.time()
    L, SL, PA = load_daily("L"), load_daily("SL"), load_daily("PA")
    panel = H.get_panel(step_days=7, horizon_days=7, target_mag=6.0)

    if STAGE == "check":
        f = make_feats(panel, L)
        # verify identical to the harness helper for one bin
        a = H.align_daily_series(L[10], panel, window_days=30, agg="mean", lag_days=1, name="a")
        mine = f[f.bin_id == 10].set_index("t_start")["x_L30"]
        print("max abs diff vs align_daily_series:", np.nanmax(np.abs(a.set_index("t_start")["a"] - mine)))
        print("NaN frac train/test:", f.merge(panel[["bin_id", "t_start", "split"]]).groupby("split")[["x_dL7", "x_L30"]]
              .apply(lambda x: x.isna().mean()))
        print(f.describe())
        print("SL std(hPa) by bin:", SL.loc["1976":].std().round(2).to_dict())
        print("PA std(hPa) by bin:", PA.loc["1976":].std().round(2).to_dict())
        # TRAIN-ONLY look: correlation of features with y_main in train
        m = panel[panel.split == "train"][["bin_id", "t_start", "y_main", "y_all"]].merge(f)
        print("train corr:", m[["y_main", "y_all", "x_dL7", "x_L30"]].corr().iloc[:2].round(4))

    elif STAGE == "primary":
        f = make_feats(panel, L)
        res, _ = ev(f, "y_main", panel)
        print(H.summarize(res))
        sur = H.surrogate_test(f, n=NSUR, mode="phase_randomize", target="y_main", panel=panel, seed=11, n_jobs=2)
        print(clean(sur))
        save("primary", {"eval": res, "surrogate": clean(sur), "null": sur["null"], "elapsed": time.time() - t0})

    elif STAGE == "secondary":
        out = {}
        f = make_feats(panel, L)
        for tgt in ("y_all", "y_main"):
            res, _ = ev(f, tgt, panel)
            print(tgt, H.summarize(res))
            o = {"eval": res}
            if tgt == "y_all":
                o["phase"] = clean(H.surrogate_test(f, n=NSUR, mode="phase_randomize", target=tgt, panel=panel, seed=21, n_jobs=2))
            o["bin_permute"] = clean(H.surrogate_test(f, n=NSUR, mode="bin_permute", target=tgt, panel=panel, seed=22, n_jobs=2))
            o["circ"] = clean(H.surrogate_test(f, n=NSUR, mode="circular_shift", target=tgt, panel=panel, seed=23, n_jobs=2))
            print(tgt, {k: (v["p_value"] if "p_value" in v else None) for k, v in o.items() if k != "eval"})
            o["gbm"] = clean(H.evaluate(f, target=tgt, model="gbm", panel=panel))
            out[f"w6_{tgt}"] = o
        # components
        for nm, dly in (("SLonly", SL), ("PAonly", PA)):
            fc = make_feats(panel, dly)
            for tgt in ("y_all", "y_main"):
                res, _ = ev(fc, tgt, panel)
                s = clean(H.surrogate_test(fc, n=NSUR, mode="phase_randomize", target=tgt, panel=panel, seed=31, n_jobs=2))
                print(nm, tgt, H.summarize(res), "p=", s["p_value"])
                out[f"w6_{nm}_{tgt}"] = {"eval": res, "phase": s}
        # M5.5 weekly
        p55 = H.get_panel(step_days=7, horizon_days=7, target_mag=5.5)
        f55 = make_feats(p55, L)
        for tgt in ("y_all", "y_main"):
            res, _ = ev(f55, tgt, p55)
            s = clean(H.surrogate_test(f55, n=NSUR, mode="phase_randomize", target=tgt, panel=p55, seed=41, n_jobs=2))
            print("M5.5", tgt, H.summarize(res), "p=", s["p_value"])
            out[f"w55_{tgt}"] = {"eval": res, "phase": s}
        save("secondary", out)

    elif STAGE == "robust":
        out = {}
        f = make_feats(panel, L)
        # leakage mask after M>=7
        fm, frac = mask_after_big(panel, f)
        for tgt in ("y_all", "y_main"):
            res, _ = ev(fm, tgt, panel)
            s = clean(H.surrogate_test(fm, n=NSUR, mode="phase_randomize", target=tgt, panel=panel, seed=51, n_jobs=2))
            print("masked", tgt, round(frac, 3), H.summarize(res), "p=", s["p_value"])
            out[f"masked_{tgt}"] = {"eval": res, "phase": s, "masked_frac": frac}
        # 30-day panel and daily panel, M6
        for (st, hz) in ((30, 30), (1, 1)):
            pp = H.get_panel(step_days=st, horizon_days=hz, target_mag=6.0)
            fp = make_feats(pp, L)
            for tgt in ("y_all", "y_main"):
                res, _ = ev(fp, tgt, pp)
                nn = NSUR if st > 1 else max(50, NSUR // 4)
                s = clean(H.surrogate_test(fp, n=nn, mode="phase_randomize", target=tgt, panel=pp, seed=61, n_jobs=2))
                print(f"s{st}h{hz}", tgt, H.summarize(res), "p=", s["p_value"])
                out[f"s{st}_h{hz}_{tgt}"] = {"eval": res, "phase": s}
        # alternate split: train 1976-2004, test 2005-2026 (Poisson background still fitted on pre-2013 in panel)
        pa = panel.copy()
        h = pd.Timedelta(days=7)
        cut = pd.Timestamp("2005-01-01")
        pa["split"] = np.where(pa.t_start + h <= cut, "train", np.where(pa.t_start >= cut, "test", "gap"))
        for tgt in ("y_all", "y_main"):
            res, _ = ev(f, tgt, pa)
            s = clean(H.surrogate_test(f, n=NSUR, mode="phase_randomize", target=tgt, panel=pa, seed=71, n_jobs=2))
            print("split2005", tgt, H.summarize(res), "p=", s["p_value"])
            out[f"split2005_{tgt}"] = {"eval": res, "phase": s}
        # causal Godin-detided variant (secondary, deviation from prereg daily mean; reported separately)
        if os.path.exists(os.path.join(DATA, "daily_L_fdgodin.parquet")):
            Lg = load_daily("L", "fdgodin")
            fg = make_feats(panel, Lg)
            for tgt in ("y_all", "y_main"):
                res, _ = ev(fg, tgt, panel)
                s = clean(H.surrogate_test(fg, n=NSUR, mode="phase_randomize", target=tgt, panel=panel, seed=81, n_jobs=2))
                print("godin", tgt, H.summarize(res), "p=", s["p_value"])
                out[f"godin_{tgt}"] = {"eval": res, "phase": s}
        # RQ sea-level source cross-check (if built)
        if os.path.exists(os.path.join(DATA, "daily_L_rq.parquet")):
            Lrq = load_daily("L", "rq")
            fr = make_feats(panel, Lrq)
            both = f.merge(fr, on=["t_start", "bin_id"], suffixes=("", "_rq"))
            out["fd_rq_corr"] = {c: float(both[[c, c + "_rq"]].corr().iloc[0, 1]) for c in ("x_dL7", "x_L30")}
            for tgt in ("y_all", "y_main"):
                res, _ = ev(fr, tgt, panel)
                print("RQ source", tgt, H.summarize(res))
                out[f"rq_{tgt}"] = {"eval": res}
        save("robust", out)
    print("elapsed", round(time.time() - t0, 1))
