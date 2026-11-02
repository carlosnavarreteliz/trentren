"""Causal features from a high-resolution catalog, on a footprint sub-panel of the harness panel."""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np
import pandas as pd
import harness as H

LN10 = np.log(10.0)


def sub_panel(step, horizon, M, b_lo, b_hi, end="2021-12-24", split_date=None):
    """Footprint sub-panel: bins b_lo..b_hi (original ids), relabelled 0..B-1; rows with t_start+h <= end+h.
    split_date: None keeps the harness split (2013-01-01); otherwise train = t_start+h <= split_date, test >= split_date."""
    p = H.get_panel(step, horizon, M)
    end = pd.Timestamp(end)
    s = p[(p.bin_id >= b_lo) & (p.bin_id <= b_hi) & (p.t_start + pd.Timedelta(days=horizon) <= end + pd.Timedelta(days=1))].copy()
    s["orig_bin"] = s.bin_id.astype(int)
    s["bin_id"] = (s.bin_id - b_lo).astype(p.bin_id.dtype)
    if split_date is not None:
        sd = pd.Timestamp(split_date)
        e = s.t_start + pd.Timedelta(days=horizon)
        s["split"] = np.where(s.t_start >= sd, "test", np.where(e <= sd, "train", "gap"))
    s = s.sort_values(["t_start", "bin_id"]).reset_index(drop=True)
    # orig_bin must not stay in the panel (the harness would treat it as a baseline feature)
    ob = s.pop("orig_bin")
    s.attrs = {}
    return s, ob.to_numpy()


def _bpos_window(m_sorted, k_idx, N, delta=0.2, min_pos=20, min_events=None):
    """b-positive over the last N events before index k (events [k-N, k)), vectorised over k_idx.
    min_events: if given, use up to N events (window = min(k, N)) and require at least min_events; otherwise exactly N."""
    if len(m_sorted) < 2:
        return np.full(len(k_idx), np.nan)
    d = np.diff(m_sorted, prepend=np.nan)
    ind = (d >= delta).astype(float)
    ind[0] = 0
    dv = np.where(ind > 0, d, 0.0)
    cI = np.r_[0, np.cumsum(ind)]
    cD = np.r_[0, np.cumsum(dv)]
    if min_events is None:
        lo = k_idx - N + 1  # first difference index inside window (pairs fully inside)
        ok = lo >= 1
    else:
        lo = np.maximum(k_idx - N + 1, 1)
        ok = k_idx >= min_events
    lo_c = np.clip(lo, 0, len(m_sorted))
    nI = cI[k_idx] - cI[lo_c]
    sD = cD[k_idx] - cD[lo_c]
    with np.errstate(divide="ignore", invalid="ignore"):
        meanexc = sD / nI - delta
        b = 1.0 / (LN10 * meanexc)
    b[(~ok) | (nI < min_pos) | ~np.isfinite(b)] = np.nan
    return b


def build_features(cat, panel, orig_bins, mthr, ref_days, start, windows=(7, 30), nb_bpos=1,
                   n_short=150, n_long=1500, prefix="x", with_b=True, depth_max=None,
                   minpos_short=20, minpos_long=100, minev_long=300):
    """cat: DataFrame time, latitude, mag, depth. panel: sub-panel (relabelled), orig_bins: original bin id per row.
    Features are NaN for t_start < start (warm-up)."""
    c = cat[cat.mag >= mthr]
    if depth_max is not None:
        c = c[c.depth <= depth_max]
    et = H.to_days(c.time)
    eb = H.lat_to_bin(c.latitude.to_numpy())
    em = c.mag.to_numpy(float)
    ts = pd.DatetimeIndex(np.sort(panel.t_start.unique()))
    ti = H.to_days(ts)
    bins = np.unique(orig_bins)
    B = len(bins)
    T = len(ti)
    out = {}
    for w in windows:
        out[f"{prefix}_act{w}"] = np.full((T, B), np.nan)
    if with_b:
        out[f"{prefix}_dbpos"] = np.full((T, B), np.nan)
    for j, b in enumerate(bins):
        tb = np.sort(et[eb == b])
        for w in windows:
            n_w = np.searchsorted(tb, ti, "left") - np.searchsorted(tb, ti - w, "left")
            n_ref = np.searchsorted(tb, ti - w, "left") - np.searchsorted(tb, ti - w - ref_days, "left")
            R = n_ref / ref_days
            out[f"{prefix}_act{w}"][:, j] = np.log((n_w + 0.5) / (w * R + 0.5))
        if with_b:
            sel = np.abs(eb - b) <= nb_bpos
            o = np.argsort(et[sel], kind="stable")
            t2, m2 = et[sel][o], em[sel][o]
            k = np.searchsorted(t2, ti, "left")
            bs = _bpos_window(m2, k, n_short, min_pos=minpos_short)
            bl = _bpos_window(m2, k, n_long, min_pos=minpos_long, min_events=minev_long)
            out[f"{prefix}_dbpos"][:, j] = bs - bl
    valid = ts >= pd.Timestamp(start)
    for k_ in out:
        out[k_][~valid, :] = np.nan
    Tm, Bm = np.meshgrid(np.arange(T), np.arange(B), indexing="ij")
    df = pd.DataFrame({"bin_id": Bm.ravel(), "t_start": ts[Tm.ravel()]})
    for k_, v in out.items():
        df[k_] = v.ravel()
    df["bin_id"] = df.bin_id.astype(panel.bin_id.dtype)
    return df


def detrend(df, cols, weeks=52):
    """Causal: feature minus its trailing mean over the previous `weeks` rows of the same bin (shift(1))."""
    d = df.sort_values(["bin_id", "t_start"]).copy()
    for c in cols:
        d[c] = d[c] - d.groupby("bin_id")[c].transform(lambda s: s.rolling(weeks, min_periods=10).mean().shift(1))
    return d.sort_values(["t_start", "bin_id"]).reset_index(drop=True)


def window_builder(feats, panel, start, end=None):
    """Surrogate builder for features that exist only inside a time window [start, end] of a longer panel.
    Applies H.ft_surrogate (same phases for all bins and columns) to the valid window only, so the surrogate has the
    same spectrum and amplitude as the real features inside the window (the harness default would phase-randomize
    the whole 1976-2021 series, most of which is an interpolated constant). builder(None) returns the real features."""
    cols = [c for c in feats.columns if c not in ("bin_id", "t_start")]
    G, cols = H._full_grid(panel, feats, cols)
    ts = np.sort(panel.t_start.unique())
    v = ts >= np.datetime64(pd.Timestamp(start))
    if end is not None:
        v &= ts <= np.datetime64(pd.Timestamp(end))

    def builder(rng):
        if rng is None:
            return H._grid_to_frame(panel, G, cols)
        S = G.copy()
        S[v] = H.ft_surrogate(G[v], rng)
        return H._grid_to_frame(panel, S, cols)
    return builder
