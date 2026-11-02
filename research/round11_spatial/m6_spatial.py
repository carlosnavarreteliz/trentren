"""M6: espacial suave + profundidad vs v3. Vease PREREG_M6_SPATIAL.md."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import forecast_short as F

RNG = np.random.default_rng(23)
CENTERS = [(la, lo) for la in np.linspace(-44, -18, 7) for lo in np.linspace(-75, -68, 4)]


def spatial(df, dep):
    lat, lon = df.lat.to_numpy(), df.lon.to_numpy()
    z60 = (dep <= 60).astype(float)
    S = {"s_lat": lat / 10.0, "s_lon": lon / 10.0,
         "s_sinlat": np.sin(np.radians(lat)), "s_coslat": np.cos(np.radians(lat)),
         "s_magz": df.x2.to_numpy() * z60, "s_latz": (lat / 10.0) * z60,
         "s_dep": np.clip(dep, 0, 600) / 150.0}
    for i, (a, b) in enumerate(CENTERS):
        S[f"s_r{i}"] = np.exp(-(((lat - a) ** 2 + (lon - b) ** 2) / (2 * 3.0 ** 2)))
    return pd.DataFrame(S, index=df.index)


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def fit_pred(tr, ca, te, cols, ycol, kind):
    if kind == "logit":
        sc = StandardScaler().fit(tr[cols].to_numpy())
        clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(tr[cols].to_numpy()), tr[ycol].to_numpy())
        s = clf.decision_function(sc.transform(ca[cols].to_numpy())).reshape(-1, 1)
        pl = LogisticRegression(C=1e6).fit(s, ca[ycol].to_numpy())
        return pl.predict_proba(clf.decision_function(sc.transform(te[cols].to_numpy())).reshape(-1, 1))[:, 1]
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=60, l2_regularization=1.0, random_state=23)
    m.fit(tr[cols].to_numpy(), tr[ycol].to_numpy())
    s = m.predict_proba(ca[cols].to_numpy())[:, 1]
    pl = LogisticRegression(C=1e6).fit(np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))).reshape(-1, 1), ca[ycol].to_numpy())
    pt = m.predict_proba(te[cols].to_numpy())[:, 1]
    return pl.predict_proba(np.log(np.clip(pt, 1e-6, 1 - 1e-6) / (1 - np.clip(pt, 1e-6, 1 - 1e-6))).reshape(-1, 1))[:, 1]


def main():
    d = F.load_local()
    df = F.build_dataset(d)
    tidx = d[(d.mag >= 4.7) & (d.time >= "1990-01-01")].index.to_numpy()
    assert len(tidx) == len(df)
    df["dep"] = d.depth.to_numpy()[tidx]
    df["zone"] = [F.zone_of(v) for v in df.lat]
    SP = spatial(df, df.dep.to_numpy())
    df = pd.concat([df.reset_index(drop=True), SP.reset_index(drop=True)], axis=1)
    cols_sp = F.COLS_V2 + list(SP.columns)
    frozen = pd.read_csv("benchmark/t1_test.csv")
    tr = df[df.time < "2021-01-01"].reset_index(drop=True)
    ca = df[(df.time >= "2021-01-01") & (df.time < "2024-01-01")].reset_index(drop=True)
    te = df[df.time >= "2024-01-01"].reset_index(drop=True)
    assert len(te) == len(frozen)
    for h in (60, 1440):
        yc = f"y_{h}"
        y = te[yc].to_numpy()
        psota = frozen[f"p_sota_{h}"].to_numpy()
        for kind in ("logit", "hgbt"):
            p = fit_pred(tr, ca, te, cols_sp, yc, kind)
            ig = (ll(y, p) - ll(y, psota)).sum() / y.sum()
            yrs = te.time.dt.year.to_numpy()
            dll = ll(y, p) - ll(y, psota)
            boots = []
            uy = np.unique(yrs)
            for _ in range(2000):
                ys = RNG.choice(uy, size=len(uy), replace=True)
                m = np.isin(yrs, ys)
                if y[m].sum() > 0:
                    boots.append(dll[m].sum() / y[m].sum())
            boots = np.array(boots)
            print(f"H={h} {kind}: IG vs v3 = {ig:+.4f} IC95% [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]")
    # leave-one-zone-out (H=60, logit)
    print("\nleave-one-zone-out (H=60, logit espacial):")
    for z in te.zone.value_counts().head(3).index:
        trm = df[(df.time < "2024-01-01") & (df.zone != z)].reset_index(drop=True)
        # recalibrar Platt dentro de trm (ult. 20% temporal)
        cut = trm.time.quantile(0.85)
        a_ = trm[trm.time <= cut].reset_index(drop=True)
        b_ = trm[trm.time > cut].reset_index(drop=True)
        tez = te[te.zone == z].reset_index(drop=True)
        p = fit_pred(a_, b_, tez, cols_sp, "y_60", "logit")
        ps = frozen.loc[te.zone.to_numpy() == z, "p_sota_60"].to_numpy()
        y = tez.y_60.to_numpy()
        ig = (ll(y, p) - ll(y, ps)).sum() / max(y.sum(), 1) if len(y) else float("nan")
        print(f"  sin {z}: test_zona n={len(tez)} pos={int(y.sum())} IG vs v3 = {ig:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
