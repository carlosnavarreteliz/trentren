"""Shared helpers for structural_static (static per-bin features + spatial surrogates)."""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np, pandas as pd
import harness as H

PRIMARY = ["x_tpga", "x_rough", "x_logsed", "x_logdist_ridge"]
SECONDARY = ["x_age", "x_slabdip", "x_trenchdepth", "x_inplate_bathy", "x_forearc_width"]
REGION = (-46.0, -17.0)


def load_profiles(csv):
    p = pd.read_csv(csv)
    return p


def bin_profile(lat, vals, bin_deg=1.0):
    """Average profile values (rows = profile points) inside each harness bin. Returns [nb, F]."""
    edges = np.arange(H.LAT_MIN, H.LAT_MAX + 1e-9, bin_deg)
    nb = len(edges) - 1
    b = np.floor((lat - H.LAT_MIN) / bin_deg).astype(int)
    out = np.full((nb, vals.shape[1]), np.nan)
    for k in range(nb):
        m = b == k
        if m.any():
            out[k] = np.nanmean(vals[m], axis=0)
    return out


def static_frame(panel, binvals, cols):
    df = panel[["bin_id", "t_start"]].copy()
    bid = panel.bin_id.to_numpy().astype(int)
    for j, c in enumerate(cols):
        df[c] = binvals[bid, j]
    return df


def spatial_ft(vals, rng):
    """multivariate FT phase-randomized surrogate along latitude (levels), same phases for all columns"""
    return H.ft_surrogate(vals, rng, method="levels")


def spatial_shift(vals, rng, min_pts=40):
    n = len(vals)
    k = int(rng.integers(min_pts, n - min_pts + 1))
    v = np.roll(vals, k, axis=0)
    if rng.random() < 0.5:
        v = v[::-1]
    return v


def run_null(panel, lat, vals, cols, target, model, n, kind, seed, bin_deg=1.0, evaluate=None):
    """Null distribution of IG for static profile features under a spatial surrogate of the profile."""
    rng = np.random.default_rng(seed)
    ev = evaluate or (lambda f: H.evaluate(f, target=target, model=model, panel=panel)["ig_per_event"])
    null = np.empty(n)
    for i in range(n):
        sv = spatial_ft(vals, rng) if kind == "ft" else spatial_shift(vals, rng)
        null[i] = ev(static_frame(panel, bin_profile(lat, sv, bin_deg), cols))
    return null


def pval(obs, null):
    null = np.asarray(null)
    p = (1 + np.sum(null >= obs)) / (len(null) + 1)
    return {"ig_obs": float(obs), "p_value": float(p), "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_q95": float(np.quantile(null, 0.95)), "n": int(len(null)),
            "significant_0.05": bool(p < 0.05), "significant_bonferroni": bool(p < H.ALPHA_BONF)}


def resplit_panel(panel, split_date="2005-01-01", test_end="2013-01-01"):
    """Copy of a harness panel with an earlier train/test split; background-rate columns recomputed from the new
    training period only (replicates harness.build_panel logic)."""
    P = panel.copy()
    S, E = pd.Timestamp(split_date), pd.Timestamp(test_end)
    h = (P.t_start.iloc[P.bin_id.nunique()] - P.t_start.iloc[0]).days  # = step; horizon assumed == step
    end_t = P.t_start + pd.Timedelta(days=h)
    P["split"] = np.where((P.t_start >= S) & (end_t <= E), "test", np.where(end_t <= S, "train", "gap"))
    nb = P.bin_id.nunique()
    tr = P.split == "train"
    ev = H.get_events(min_mag=H.M_MODEL)
    ev = ev[(ev.time >= H.PANEL_START) & (ev.time < S)]
    span = (S - pd.Timestamp(H.PANEL_START)).days
    b = H.lat_to_bin(ev.latitude.to_numpy())
    rate45 = np.bincount(b[b >= 0], minlength=nb)[:nb] / span * h
    P["log_bg_m45"] = np.log10(H._smooth_bins(rate45, 1.0) + 1e-4)[P.bin_id]
    for tgt in ["all", "main"]:
        k = P[tr].groupby("bin_id")[f"y_{tgt}"].agg(["sum", "count"]).reindex(range(nb), fill_value=0)
        glob = k["sum"].sum() / k["count"].sum()
        freq = (H._smooth_bins(k["sum"].to_numpy().astype(float), 1.0) + glob) / (k["count"].to_numpy() + 1.0)
        P[f"p_poisson_{tgt}"] = np.clip(freq, 1e-6, 0.5)[P.bin_id]
    P["log_bg_target"] = np.log10(P["p_poisson_all"])
    return P
