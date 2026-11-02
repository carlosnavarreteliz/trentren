"""
Evaluate GNSS transient features through the shared harness.

usage: python -I evaluate_gnss.py <station_anoms.parquet> <out_dir> [n_surr] [stage]
  stage = primary | robust | all (default all)
"""
import sys, json, time
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import harness as H

ANOM = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
N_SURR = int(sys.argv[3]) if len(sys.argv) > 3 else 200
STAGE = sys.argv[4] if len(sys.argv) > 4 else "all"
ERA = pd.Timestamp("2007-01-01")
LON_MAX = -68.5
NB = 29

an = pd.read_parquet(ANOM)
an = an[an.lon <= LON_MAX].copy()
an["bin"] = H.lat_to_bin(an.lat.to_numpy())
an = an[(an.bin >= 0) & (an.bin < NB)]
WEEKS = pd.date_range("1976-01-01", "2026-10-17", freq="7D")


def bin_grid(lag=2, W=30, clean=False, absval=False):
    """weekly-issue x bin grid of the median station z over bins b-1..b+1; NaN where no station."""
    a = an[an.lag == lag]
    if clean:
        a = a[~a.post65]
    col = f"z{W}"
    a = a[np.isfinite(a[col])]
    Z = a.pivot_table(index="t_start", columns="sta", values=col, aggfunc="first").reindex(WEEKS)
    if absval:
        Z = Z.abs()
    sb = a.groupby("sta").bin.first().reindex(Z.columns).to_numpy()
    G = np.full((len(WEEKS), NB), np.nan)
    C = np.zeros((len(WEEKS), NB))
    Zv = Z.to_numpy()
    for b in range(NB):
        cols = np.where(np.abs(sb - b) <= 1)[0]
        if len(cols) == 0:
            continue
        sub = Zv[:, cols]
        C[:, b] = np.isfinite(sub).sum(1)
        with np.errstate(all="ignore"):
            G[:, b] = np.where(C[:, b] > 0, np.nanmedian(np.where(np.isfinite(sub), sub, np.nan), axis=1), np.nan)
    return G, C


def to_frame(panel, G, name):
    """map weekly grid to the panel issue times (latest weekly issue <= t_start; causal), fill missing with 0."""
    ts = np.sort(panel.t_start.unique())
    idx = np.searchsorted(WEEKS.values, ts, side="right") - 1
    Gp = np.where(idx[:, None] >= 0, G[np.clip(idx, 0, None)], np.nan)
    df = pd.DataFrame({"t_start": np.repeat(ts, NB), "bin_id": np.tile(np.arange(NB), len(ts))})
    df[name] = np.nan_to_num(Gp.ravel(), nan=0.0)
    return df, np.isfinite(Gp)


def make_builder(panel, G, name, era=ERA):
    base_df, mask = to_frame(panel, G, name)
    ts = np.sort(panel.t_start.unique())
    rows = ts >= np.datetime64(era)
    Z = base_df[name].to_numpy().reshape(len(ts), NB)

    def builder(rng):
        if rng is None:
            return base_df
        Y = Z.copy()
        S = H.ft_surrogate(Z[rows], rng, method="levels")
        S[~mask[rows]] = 0.0
        Y[rows] = S
        d = base_df.copy(); d[name] = Y.ravel()
        return d
    return builder, base_df


def restrict(panel, start=ERA):
    return panel[panel.t_start >= start].reset_index(drop=True)


def resplit(panel, test_start):
    p = panel.copy()
    h = pd.Timedelta(days=int((p.t_start.iloc[NB] - p.t_start.iloc[0]).days))
    test_start = pd.Timestamp(test_start)
    p["split"] = np.where(p.t_start >= test_start, "test", np.where(p.t_start + h <= test_start, "train", "gap"))
    return p


def clean_res(r):
    return {k: v for k, v in r.items() if k != "preds"}


def run(panel, G, name, target, n_surr=N_SURR, model="logit", modes=("phase_randomize",), seed=0, tag=""):
    builder, df = make_builder(panel, G, name)
    r = H.evaluate(df, target=target, model=model, panel=panel, return_preds=True)
    bb = H.block_bootstrap_ci(r["preds"])
    out = {"tag": tag, "feature": name, "target": target, "model": model, "res": clean_res(r), "bootstrap": bb,
           "ig_per_event": r["ig_per_event"], "ig_full_vs_etas": r["ig_full_vs_etas"],
           "auc_base": r["base"]["roc_auc"], "auc_full": r["full"]["roc_auc"],
           "ass_base": r["base"]["molchan_ass"], "ass_full": r["full"]["molchan_ass"],
           "n_test_events": r["n_test_events"], "n_train_events": r["n_train_events"]}
    t0 = time.time()
    for m in modes:
        if n_surr <= 0 or model != "logit":
            break
        if m == "phase_randomize":
            s = H.surrogate_test(builder=builder, n=n_surr, mode=m, target=target, panel=panel, seed=seed, n_jobs=2)
        else:
            s = H.surrogate_test(df, n=n_surr, mode=m, target=target, panel=panel, seed=seed, n_jobs=2)
        out[f"sur_{m}"] = {k: v for k, v in s.items() if k != "null"}
    print(f"[{tag}] {name} {target} {model}: IG={r['ig_per_event']:+.4f} vsETAS={r['ig_full_vs_etas']:+.4f} "
          f"AUC {r['base']['roc_auc']:.3f}->{r['full']['roc_auc']:.3f} coef={r.get('coef_extra')} "
          f"boot={bb} " + " ".join(f"{m}:p={out[f'sur_{m}']['p_value']:.4f}" for m in modes if f'sur_{m}' in out)
          + f" ({time.time()-t0:.0f}s)", flush=True)
    return out, r


def main():
    results = []
    pw = H.get_panel(7, 7, 6.0)
    pr = restrict(pw)
    G30, C30 = bin_grid(2, 30)
    if STAGE in ("primary", "all"):
        # coverage diagnostics
        cov = pd.DataFrame({"year": WEEKS.year, "bins_cov": (C30 > 0).sum(1), "mean_sta": C30.mean(1)})
        print(cov.groupby("year").mean().round(1).loc[2000:].to_string())
        # PRIMARY: restricted panel, y_main
        o, r = run(pr, G30, "x_gtw_z30", "y_main", tag="PRIMARY restricted2007 weekly M6")
        results.append(o)
        r["preds"].to_parquet(OUT / "preds_primary_ymain.parquet")
        o, r = run(pr, G30, "x_gtw_z30", "y_all", tag="restricted2007 weekly M6")
        results.append(o)
        r["preds"].to_parquet(OUT / "preds_primary_yall.parquet")
        json.dump(results, open(OUT / "results_primary.json", "w"), indent=1, default=float)
    if STAGE in ("robust", "all"):
        nr = max(100, N_SURR // 2)
        # full panel
        for tgt in ("y_main", "y_all"):
            o, _ = run(pw, G30, "x_gtw_z30", tgt, n_surr=nr, tag="full1976 weekly M6"); results.append(o)
        # other surrogate modes on primary config
        for tgt in ("y_main", "y_all"):
            o, _ = run(pr, G30, "x_gtw_z30", tgt, n_surr=nr, modes=("bin_permute", "circular_shift"),
                       tag="restricted2007 weekly M6 altsur"); results.append(o)
        # GBM
        for tgt in ("y_main", "y_all"):
            o, _ = run(pr, G30, "x_gtw_z30", tgt, n_surr=0, model="gbm", tag="restricted2007 weekly M6 GBM"); results.append(o)
        # secondary features
        variants = {"x_gtw_z90": bin_grid(2, 90)[0], "x_gtw_absz30": bin_grid(2, 30, absval=True)[0],
                    "x_gtw_z30_clean": bin_grid(2, 30, clean=True)[0], "x_gtw_z30_lag14": bin_grid(14, 30)[0]}
        for nm, G in variants.items():
            for tgt in ("y_main", "y_all"):
                o, _ = run(pr, G, nm, tgt, n_surr=nr, tag="restricted2007 weekly M6 variant"); results.append(o)
        # M5.5 weekly
        p55 = restrict(H.get_panel(7, 7, 5.5))
        for tgt in ("y_main", "y_all"):
            o, _ = run(p55, G30, "x_gtw_z30", tgt, n_surr=nr, tag="restricted2007 weekly M5.5"); results.append(o)
        # 30-day panel M6
        p30 = restrict(H.get_panel(30, 30, 6.0))
        for tgt in ("y_main", "y_all"):
            o, _ = run(p30, G30, "x_gtw_z30", tgt, n_surr=nr, tag="restricted2007 30d M6"); results.append(o)
        p30b = restrict(H.get_panel(30, 30, 6.0))
        o, _ = run(p30b, bin_grid(2, 90)[0], "x_gtw_z90", "y_main", n_surr=nr, tag="restricted2007 30d M6 z90"); results.append(o)
        # alternate split: test 2016+
        ps = resplit(pr, "2016-01-01")
        for tgt in ("y_main", "y_all"):
            o, _ = run(ps, G30, "x_gtw_z30", tgt, n_surr=nr, tag="restricted2007 weekly M6 split2016"); results.append(o)
        json.dump(results, open(OUT / f"results_{STAGE}.json", "w"), indent=1, default=float)

        # leakage check: correlation of feature with station count, and with recent big quakes
        df, _ = to_frame(pr, G30, "x")
        _, Cm = to_frame(pr, C30, "n")
        dfc, _ = to_frame(pr, C30, "nsta")
        m = pr[["bin_id", "t_start", "maxmag_nbr_365d", "log_days_since_m6_nbr", "cnt_nbr_30d"]].merge(df, on=["bin_id", "t_start"]).merge(dfc, on=["bin_id", "t_start"])
        print("corr with station count / clustering features:\n", m[["x", "nsta", "maxmag_nbr_365d", "log_days_since_m6_nbr", "cnt_nbr_30d"]].corr(method="spearman").round(3))


if __name__ == "__main__":
    main()
