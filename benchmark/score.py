"""Puntua un envio T1 contra el test congelado. Uso: score.py envio.csv [--h 60]."""
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

TEST = "benchmark/t1_test.csv"
RNG = np.random.default_rng(99)


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--h", type=int, default=60, choices=[60, 180, 1440])
    ap.add_argument("--thr", type=float, default=None, help="umbral alerta (default: mediana sota)")
    a = ap.parse_args()
    t = pd.read_csv(TEST, parse_dates=["time"])
    s = pd.read_csv(a.csv)
    m = t.merge(s, on="trigger_idx", how="inner", validate="one_to_one")
    if len(m) != len(t):
        print(f"ERROR: envio cubre {len(m)}/{len(t)} triggers", file=sys.stderr)
        sys.exit(1)
    y = m[f"y_{a.h}"].to_numpy()
    p = m[f"p_{a.h}"].to_numpy()
    ps = m[f"p_sota_{a.h}"].to_numpy()
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        print("ERROR: probabilidades invalidas", file=sys.stderr)
        sys.exit(1)
    ig = (ll(y, p) - ll(y, ps)).sum() / y.sum()
    yrs = m.time.dt.year.to_numpy()
    dll = ll(y, p) - ll(y, ps)
    boots = []
    for uy in [np.unique(yrs)] * 2000:
        ys = RNG.choice(uy, size=len(uy), replace=True)
        mk = np.isin(yrs, ys)
        if y[mk].sum() > 0:
            boots.append(dll[mk].sum() / y[mk].sum())
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    thr = a.thr if a.thr is not None else float(np.median(ps))
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    print(f"T1 H={a.h}min n={len(m)} pos={int(y.sum())}")
    print(f"IG vs SOTA = {ig:+.4f} IC95% [{lo:+.4f}, {hi:+.4f}] -> "
          f"{'MEJORA' if lo > 0 else 'no mejora'}")
    print(f"AUC={roc_auc_score(y, p):.3f} (sota {roc_auc_score(y, ps):.3f}) "
          f"PR-AUC={average_precision_score(y, p):.3f} Brier={brier_score_loss(y, p):.4f}")
    print(f"alertas@{thr:.3f}: {int(pred.sum())} aciertos={tp} "
          f"P={tp / max(pred.sum(), 1):.2f} R={tp / max(y.sum(), 1):.2f}")


if __name__ == "__main__":
    sys.exit(main())
