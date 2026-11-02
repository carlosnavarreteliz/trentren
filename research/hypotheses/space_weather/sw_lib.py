"""Shared helpers for the space_weather hypothesis (features + surrogate builders)."""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np
import pandas as pd
import harness as H


def load_daily(path):
    d = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    idx = pd.date_range("1970-01-01", "2026-10-17", freq="D")
    d = d.reindex(idx)
    d["lognp"] = np.log10(d["np"])
    d["logf107"] = np.log10(d["f107"])
    return d


def aaft(vals, rng):
    """Amplitude-adjusted FT surrogate: harness ft_surrogate (same spectrum) then rank-remap to the original
    marginal distribution on the finite positions (keeps heavy tails, e.g. Dst storms). rng=None -> original."""
    if rng is None:
        return vals
    sur = H.ft_surrogate(vals, rng)
    ok = np.isfinite(vals) & np.isfinite(sur)
    out = np.full_like(vals, np.nan)
    srt = np.sort(vals[ok])
    ranks = np.argsort(np.argsort(sur[ok]))
    out[ok] = srt[ranks]
    return out


def causal_diff(s, w):
    """trailing w-day mean minus the trailing w-day mean ending w days earlier (strictly causal)."""
    m = s.rolling(w, min_periods=int(0.6 * w)).mean()
    return m - m.shift(w)


# feature spec: name -> (raw column, transform of the raw daily series, align kwargs)
FAMILY_B = {
    "x_np7": ("lognp", None, dict(window_days=7, agg="mean", lag_days=1)),
    "x_v7": ("v", None, dict(window_days=7, agg="mean", lag_days=1)),
    "x_bzs7": ("bzs", None, dict(window_days=7, agg="mean", lag_days=1)),
    "x_kpmax7": ("kp_max", None, dict(window_days=7, agg="max", lag_days=1)),
    "x_dstmin7": ("dst_min", None, dict(window_days=7, agg="min", lag_days=1)),
    "x_storm7": ("dst_min", "storm", dict(window_days=7, agg="max", lag_days=1)),
    "x_f107_d81": ("logf107", "diff81", dict(window_days=1, agg="last", lag_days=1)),
    "x_ssn_d81": ("ssn", "diff81", dict(window_days=1, agg="last", lag_days=1)),
}


def make_feature(daily, name, panel, raw_vals=None, spec=None):
    col, tr, kw = spec if spec is not None else FAMILY_B[name]
    v = daily[col].to_numpy(float) if raw_vals is None else raw_vals
    s = pd.Series(v, index=daily.index)
    if tr == "storm":
        s = (s <= -50).astype(float).where(s.notna())
    elif tr == "diff81":
        s = causal_diff(s, 81)
    return H.align_daily_series(s, panel, name=name, **kw)


def builder_for(daily, name, panel, spec=None, kind="ft"):
    col = (spec if spec is not None else FAMILY_B[name])[0]
    raw = daily[col].to_numpy(float)

    if kind == "ft":
        def b(rng):
            return make_feature(daily, name, panel, raw_vals=aaft(raw, rng), spec=spec)
    else:  # end-matched circular shift of the raw daily series
        def b(shift_days):
            # NaN gaps move with the data (a true circular shift of the raw series)
            v = raw if shift_days == 0 else H.circular_shift_series(pd.Series(raw), shift_days).to_numpy()
            return make_feature(daily, name, panel, raw_vals=v, spec=spec)
    return b


def resplit(panel, train_end, test_start, test_end=None, train_start=None):
    """Copy of panel with a different train/test split (purged)."""
    p = panel.copy()
    ts = np.sort(p.t_start.unique())
    h = p.attrs.get("horizon_days")
    if h is None:
        h = int((ts[1] - ts[0]) / np.timedelta64(1, "D"))
    te = p.t_start + pd.Timedelta(days=h)
    split = np.full(len(p), "gap", dtype=object)
    tr = te <= pd.Timestamp(train_end)
    if train_start is not None:
        tr &= p.t_start >= pd.Timestamp(train_start)
    tst = p.t_start >= pd.Timestamp(test_start)
    if test_end is not None:
        tst &= te <= pd.Timestamp(test_end)
    split[tr.to_numpy()] = "train"
    split[tst.to_numpy()] = "test"
    p["split"] = split
    return p
