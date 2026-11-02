"""shared helpers for catalog_metadata_search (selection + final test)."""
import sys
sys.path.insert(0, "/Users/cnavarreteliz/Documents/Repos/earthquakes/research")
import numpy as np
import pandas as pd
import harness as H

FOLDS_2013 = [("1995-01-01", "2002-01-01"), ("2002-01-01", "2008-01-01"), ("2008-01-01", "2013-01-01")]
FOLDS_2006 = [("1990-01-01", "1995-01-01"), ("1995-01-01", "2000-01-01"), ("2000-01-01", "2006-01-01")]

ARTIFACT_KEYS = ("_m_", "_s_", "_a_fmw", "_a_fmb", "_a_fml", "_a_fixdep", "_a_night", "_a_weekend", "_a_hourR",
                 "_c_rms", "_w_s_")


def is_artifact_prone(name):
    return any(k in name for k in ARTIFACT_KEYS)


def family(name):
    p = name.split("_")
    return p[2]  # a, m, s, c, w, sp


def aligned_matrix(F, panel, cols):
    """F (weekly grid, keyed bin_id,t_start) -> array aligned to panel rows (as-of backward if grids differ)."""
    tp = np.sort(panel.t_start.unique())
    tg = np.sort(F.t_start.unique())
    if len(tp) == len(tg) and (tp == tg).all():
        G = F
    else:
        import features as FL
        G = FL.align_asof(F, panel)
    left = panel[["bin_id", "t_start"]].copy()
    left["bin_id"] = left.bin_id.astype(int)
    G = G.copy()
    G["bin_id"] = G.bin_id.astype(int)
    m = left.merge(G[["bin_id", "t_start"] + cols], on=["bin_id", "t_start"], how="left")
    return m[cols].to_numpy(dtype=float)


def winsorize(X, train_mask):
    X = X.copy()
    for j in range(X.shape[1]):
        v = X[train_mask, j]
        v = v[np.isfinite(v)]
        if len(v) < 20:
            continue
        lo, hi = np.percentile(v, [0.5, 99.5])
        X[:, j] = np.clip(X[:, j], lo, hi)
    return X


def fold_panel(panel, v_start, v_end, h_days):
    vs, ve = pd.Timestamp(v_start), pd.Timestamp(v_end)
    p = panel[panel.t_start < ve].copy()
    end_t = p.t_start + pd.Timedelta(days=h_days)
    p["split"] = np.where(end_t <= vs, "train", np.where((p.t_start >= vs) & (end_t <= ve), "test", "gap"))
    return p.reset_index(drop=True)


def frame(panel, X, cols):
    df = panel[["bin_id", "t_start"]].copy()
    for j, c in enumerate(cols):
        df[c] = X[:, j]
    return df


def bh(pvals, q):
    p = np.asarray(pvals)
    n = len(p)
    o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    sel = np.zeros(n, bool)
    sel[o[:k]] = True
    return sel


def year_boot_p(D, Y, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    k = len(D)
    idx = rng.integers(0, k, size=(n_boot, k))
    num, den = D[idx].sum(1), Y[idx].sum(1)
    ig = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return float(np.mean(ig <= 0))
