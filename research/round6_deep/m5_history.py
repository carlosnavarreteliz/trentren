"""M5: historia larga USGS, features ricas vs logit-3. Vease PREREG_M5_HISTORY.md."""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

RNG = np.random.default_rng(71)
R_KM = 6371.0


def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def bpos(seq, delta=0.1):
    d = np.diff(seq)
    d = d[d >= delta]
    if len(d) < 5:
        return np.nan
    return np.log10(np.e) / max(d.mean() - delta, 0.01)


def main():
    u = pd.read_csv("data/raw/chile_usgs_1900_2026.csv",
                    usecols=["time", "latitude", "longitude", "depth", "mag", "type"])
    u = u[u["type"] == "earthquake"]
    u["time"] = pd.to_datetime(u["time"], format="ISO8601", utc=True).dt.tz_convert(None)
    u = u[u.latitude.between(-46, -17) & u.longitude.between(-78, -66)].dropna(subset=["mag"])
    u = u.sort_values("time").reset_index(drop=True)
    u["tmin"] = (u.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
    tm, la, lo, mg = (u[c].to_numpy() for c in ["tmin", "latitude", "longitude", "mag"])
    dp = u.depth.to_numpy()
    trig = u[(u.mag >= 4.7) & (u.time >= "1990-01-01") & (u.time < "2026-10-01")].reset_index()
    rows = []
    for _, x in trig.iterrows():
        i0 = int(x["index"])
        lo365 = np.searchsorted(tm, x.tmin - 365 * 1440, side="left")
        pre = np.arange(lo365, i0)
        dd = hav_km(x.latitude, x.longitude, la[pre], lo[pre]) if len(pre) else np.array([])
        near = pre[(dd <= 150) & (mg[pre] >= 4.7)]
        if len(near) < 3:
            continue
        dt = x.tmin - tm[near]
        m7d, z = mg[near], dp[near]
        r = {"time": x.time, "b_mag": float(x.mag)}
        for wname, wmin in [("1h", 60), ("3h", 180), ("24h", 1440), ("7d", 7 * 1440),
                            ("30d", 30 * 1440), ("365d", 365 * 1440)]:
            m = dt <= wmin
            r[f"c_{wname}"] = float(np.log1p(m.sum()))
            r[f"ci_{wname}"] = float(np.log1p(np.sum(m & (z <= 60))))
            r[f"cs_{wname}"] = float(np.log1p(np.sum(m & (z > 60))))
        r["b_tot24"] = r["c_24h"]
        for wname, wmin in [("24h", 1440), ("7d", 7 * 1440), ("30d", 30 * 1440)]:
            m = dt <= wmin
            r[f"mx_{wname}"] = float(m7d[m].max()) if m.any() else 4.7
        seq = m7d[np.argsort(tm[near])]
        r["bpos20"] = bpos(seq[-20:])
        r["bpos50"] = bpos(seq[-50:])
        for p in (1.08, 1.3):
            mm = dt <= 30 * 1440
            r[f"om_{p}"] = float(np.log10(np.sum(10 ** (1.1 * (m7d[mm] - 4.7)) *
                                                 (dt[mm] / 1440.0 + 0.02) ** (-p)) + 1e-6))
        for mt in (5.0, 6.0):
            last = tm[near[m7d >= mt]]
            r[f"dsince_{mt}"] = float(np.log10(1 + (x.tmin - last.max()) / 1440.0)) if len(last) else 5.0
        z7 = z[dt <= 7 * 1440]
        r["ztrend"] = float(z7.mean() - z.mean()) if len(z7) else 0.0
        r["t_dep"] = float(x.depth / 150.0)
        r["t_z"] = 1.0 if x.depth <= 60 else 0.0
        hi = np.searchsorted(tm, x.tmin + 60, side="right")
        fut = np.arange(i0 + 1, hi)
        y = 0
        if len(fut):
            cand = fut[mg[fut] >= 5.0]
            if len(cand):
                y = int((hav_km(x.latitude, x.longitude, la[cand], lo[cand]) <= 150).any())
        r["y"] = y
        rows.append(r)
    df = pd.DataFrame(rows)
    feats = [c for c in df.columns if c not in ("time", "y")]
    print(f"triggers={len(df)} feats={len(feats)}")
    tr = df[df.time < "2018-01-01"].reset_index(drop=True)
    te = df[df.time >= "2018-01-01"].reset_index(drop=True)
    print(f"train={len(tr)} pos={tr.y.sum()} | test={len(te)} pos={te.y.sum()}")
    med = tr[feats].median().fillna(0.0)
    Xtr = tr[feats].fillna(med).to_numpy()
    Xte = te[feats].fillna(med).to_numpy()
    ytr, yte = tr.y.to_numpy(), te.y.to_numpy()
    nva = int(0.15 * len(tr))
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=80,
                                         l2_regularization=1.0, early_stopping=True,
                                         random_state=71)
    gbm.fit(Xtr[:-nva], ytr[:-nva])
    p_gbm = gbm.predict_proba(Xte)[:, 1]
    sc = StandardScaler().fit(Xtr)
    lg = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(Xtr), ytr)
    p_log = lg.predict_proba(sc.transform(Xte))[:, 1]
    bcols = ["b_mag", "c_1h", "b_tot24"]
    scb = StandardScaler().fit(tr[bcols].fillna(med).to_numpy())
    lb = LogisticRegression(C=1.0).fit(scb.transform(tr[bcols].fillna(med).to_numpy()), ytr)
    p_base = lb.predict_proba(scb.transform(te[bcols].fillna(med).to_numpy()))[:, 1]

    def ll(y, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return y * np.log(p) + (1 - y) * np.log1p(-p)

    for name, p in [("GBM", p_gbm), ("logit-rico", p_log)]:
        ig = (ll(yte, p) - ll(yte, p_base)).sum() / yte.sum()
        print(f"IG {name} vs logit-3 = {ig:+.4f} | AUC={roc_auc_score(yte, p):.3f} "
              f"(base {roc_auc_score(yte, p_base):.3f}) PR={average_precision_score(yte, p):.3f} "
              f"(base {yte.mean():.3f})")
        yrs = te.time.dt.year.to_numpy()
        uy = np.unique(yrs)
        dll = ll(yte, p) - ll(yte, p_base)
        boots = []
        for _ in range(2000):
            ys = RNG.choice(uy, size=len(uy), replace=True)
            m = np.isin(yrs, ys)
            if yte[m].sum() > 0:
                boots.append(dll[m].sum() / yte[m].sum())
        boots = np.array(boots)
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        cnt = 0
        for _ in range(5000):
            yp = RNG.permutation(yte)
            cnt += yp.sum() > 0 and ((ll(yp, p) - ll(yp, p_base)).sum() / yp.sum()) >= ig
        print(f"  IC95% [{lo:+.4f}, {hi:+.4f}] p={(1 + cnt) / 5001:.4f}")


if __name__ == "__main__":
    sys.exit(main())
