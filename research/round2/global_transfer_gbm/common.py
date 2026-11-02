"""Shared helpers: exact year-block sign-flip test, metrics, level-matched IG."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import brentq


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def year_diffs(y, pa, pb, year):
    d = ll(y, pa) - ll(y, pb)
    g = pd.DataFrame({"year": year, "d": d}).groupby("year").d.sum()
    return g.to_numpy(), g.index.to_numpy()


def signflip_p(y, pa, pb, year):
    """Exact paired year-block sign-flip randomization test, one-sided (A better than B).
    Returns (p, IG per event, per-year diffs)."""
    D, yrs = year_diffs(y, pa, pb, year)
    K = len(D)
    S = ((np.arange(2 ** K)[:, None] >> np.arange(K)) & 1) * 2.0 - 1.0
    stats = S @ D
    obs = D.sum()
    p = float(np.mean(stats >= obs - 1e-9))
    N = max(int(np.sum(y)), 1)
    return p, float(obs / N), dict(zip([int(v) for v in yrs], np.round(D, 3).tolist()))


def level_match(y, p):
    """Oracle diagnostic: shift p by one logit constant so that sum(p) == sum(y) on the evaluated set."""
    z = logit(np.clip(p, 1e-9, 1 - 1e-9))
    target = float(np.sum(y))
    f = lambda c: expit(z + c).sum() - target
    c = brentq(f, -20, 20)
    return expit(z + c)
