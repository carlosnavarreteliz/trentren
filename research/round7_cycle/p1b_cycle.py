"""P1b: renovacion por segmento + energia acumulada. Vease PREREG_P1B_CYCLE.md."""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ZONES = [("Arica", -19.5, -17.0), ("Iquique", -21.5, -19.5),
         ("Antofagasta", -25.0, -21.5), ("Atacama", -29.0, -25.0),
         ("Coquimbo", -32.2, -29.0), ("Valparaiso", -34.5, -32.2),
         ("Maule", -38.5, -34.5), ("Valdivia", -44.0, -38.5)]
RNG = np.random.default_rng(77)


def main():
    d = pd.read_csv("data/raw/chile_usgs_1900_2026.csv",
                    usecols=["time", "latitude", "longitude", "mag", "type"])
    d = d[d["type"] == "earthquake"]
    d["time"] = pd.to_datetime(d["time"], format="ISO8601", utc=True).dt.tz_convert(None)
    d = d.dropna(subset=["mag"]).sort_values("time").reset_index(drop=True)
    d["M0"] = 10 ** (1.5 * d.mag + 9.1)
    d["ben"] = 10 ** (0.75 * d.mag)
    rows = []
    for zi, (zn, a, b) in enumerate(ZONES):
        z = d[d.latitude.between(a, b)]
        big = z[z.mag >= 7.5].sort_values("time").reset_index(drop=True)
        for yr in range(1960, 2022):
            t0 = pd.Timestamp(f"{yr}-01-01")
            past_big = big[big.time < t0]
            t_last = past_big.time.max() if len(past_big) else pd.Timestamp("1904-01-01")
            since = (t0 - t_last).days / 365.25
            acc = z[(z.time >= t_last) & (z.time < t0) & (z.mag >= 5.0)]
            m20 = z[(z.time >= t0 - pd.Timedelta(days=20 * 365.25)) & (z.time < t0) & (z.mag >= 5.0)]
            fut = big[(big.time >= t0) & (big.time < t0 + pd.Timedelta(days=5 * 365.25))]
            rows.append({"zone": zn, "year": yr,
                         "f_since": float(np.log1p(since)),
                         "f_mom": float(np.log10(acc.M0.sum() + 1)),
                         "f_ben": float(np.log10(acc.ben.sum() + 1)),
                         "f_def": float(since * 66.0 / 1000.0),
                         "f_rate": float(len(m20) / 20.0),
                         "y": int(len(fut) > 0)})
    df = pd.DataFrame(rows)
    tr = df[df.year <= 1990].reset_index(drop=True)
    te = df[df.year >= 1991].reset_index(drop=True)
    print(f"train={len(tr)} pos={tr.y.sum()} | test={len(te)} pos={te.y.sum()}")
    cols = ["f_since", "f_mom", "f_ben", "f_def", "f_rate"]
    sc = StandardScaler().fit(tr[cols].to_numpy())
    clf = LogisticRegression(C=1.0).fit(sc.transform(tr[cols].to_numpy()), tr.y.to_numpy())
    p = clf.predict_proba(sc.transform(te[cols].to_numpy()))[:, 1]
    # Poisson zonal (tasas train)
    rate = tr.groupby("zone").y.mean()
    pp = te.zone.map(rate).to_numpy()
    y = te.y.to_numpy()

    def ll(v, q):
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return v * np.log(q) + (1 - v) * np.log1p(-q)

    ig = (ll(y, p) - ll(y, pp)).sum() / y.sum()
    print(f"IG vs Poisson = {ig:+.4f} | AUC modelo={roc_auc_score(y, p):.3f} poisson={roc_auc_score(y, pp):.3f}")
    print("coefs:", dict(zip(cols, np.round(clf.coef_[0], 3))))
    dll = ll(y, p) - ll(y, pp)
    keys = (te.zone + "_" + (te.year // 10 * 10).astype(str)).to_numpy()
    uk = np.unique(keys)
    boots = []
    for _ in range(2000):
        ks = RNG.choice(uk, size=len(uk), replace=True)
        m = np.isin(keys, ks)
        if y[m].sum() > 0:
            boots.append(dll[m].sum() / y[m].sum())
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    print(f"IC95% (zona-decada): [{lo:+.4f}, {hi:+.4f}]")
    print("veredicto:", "NULO" if lo <= 0 else "SEÑAL")
    print("\npor zona (test): tasa modelo media vs observado:")
    te = te.assign(p=p, pp=pp)
    print(te.groupby("zone").agg(p=("p", "mean"), pp=("pp", "mean"), obs=("y", "mean")).round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
