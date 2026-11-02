"""M4: GBDT 60+ features vs E1-logit. Vease PREREG_M4_GBDT.md."""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

CLEAN = "research/data/round2_enhanced_catalog/riedel/primary_clean.parquet"
RNG = np.random.default_rng(63)
R_KM = 6371.0


def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def bpos(dm, delta=0.2):
    d = np.diff(dm)
    d = d[d >= delta]
    if len(d) < 5:
        return np.nan
    return np.log10(np.e) / max(d.mean() - delta, 0.01)


def main():
    d = pd.read_parquet(CLEAN)
    d["time"] = pd.to_datetime(d["time"])
    d["tmin"] = (d.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
    d = d.sort_values("time").reset_index(drop=True)
    tm, la, lo, mg = (d[c].to_numpy() for c in ["tmin", "Lat", "Lon", "Ml"])
    dp = d.Dep.to_numpy()
    ty = d.Type.to_numpy()
    # tasa historica del bin (train <=2019)
    binof = lambda v: int(np.floor(v))
    tr_mask_time = d.time < "2020-01-01"
    bins, counts = np.unique(np.floor(d[tr_mask_time].Lat), return_counts=True)
    binrate = {b: c / 1095.0 for b, c in zip(bins, counts)}
    trig = d[(d.Ml >= 4.0) & (d.time >= "2017-06-01") & (d.time < "2021-11-15")].reset_index()
    rows = []
    for _, x in trig.iterrows():
        i0 = int(x["index"])
        lo30 = np.searchsorted(tm, x.tmin - 30 * 1440, side="left")
        pre = np.arange(lo30, i0)
        dd = hav_km(x.Lat, x.Lon, la[pre], lo[pre]) if len(pre) else np.array([])
        near = pre[(dd <= 150) & (mg[pre] >= 2.5)]
        if len(near) < 5:
            continue
        dt = x.tmin - tm[near]
        m7d = mg[near]
        z = dp[near]
        tyv = ty[near]
        r = {"time": x.time}
        for wname, wmin in [("10m", 10), ("1h", 60), ("6h", 360), ("24h", 1440),
                            ("7d", 7 * 1440), ("30d", 30 * 1440)]:
            m = dt <= wmin
            r[f"c_{wname}"] = float(np.log1p(m.sum()))
            r[f"ci_{wname}"] = float(np.log1p(np.sum(m & (tyv == "Interplate"))))
            r[f"cs_{wname}"] = float(np.log1p(np.sum(m & (tyv == "Intraslab"))))
            r[f"cz_{wname}"] = float(np.log1p(np.sum(m & (z <= 60))))
        for wname, wmin in [("24h", 1440), ("7d", 7 * 1440), ("30d", 30 * 1440)]:
            m = dt <= wmin
            r[f"mx_{wname}"] = float(m7d[m].max()) if m.any() else 2.5
        seq_mag = m7d[np.argsort(tm[near])][-60:]  # ultimos 60 por tiempo
        r["bpos20"] = bpos(seq_mag[-20:]) if len(seq_mag) >= 20 else np.nan
        r["bpos50"] = bpos(seq_mag[-50:]) if len(seq_mag) >= 50 else np.nan
        for p in (1.08, 1.3):
            r[f"om_{p}"] = float(np.log10(np.sum(10 ** (1.1 * (m7d[dt <= 30 * 1440] - 4.7)) *
                                                 (dt[dt <= 30 * 1440] / 1440.0 + 0.02) ** (-p)) + 1e-6))
        for mt in (5.0, 6.0):
            last = tm[near[m7d >= mt]]
            r[f"dsince_{mt}"] = float(np.log10(1 + (x.tmin - last.max()) / 1440.0)) if len(last) else 4.0
        z7 = z[dt <= 7 * 1440]
        z30 = z
        r["ztrend"] = float(z7.mean() - z30.mean()) if len(z7) and len(z30) else 0.0
        r["t_mag"], r["t_dep"] = float(x.Ml), float(x.Dep / 150.0)
        r["t_z"] = 1.0 if x.Dep <= 60 else 0.0
        r["binrate"] = float(np.log10(binrate.get(binof(x.Lat), 1.0) + 1e-6))
        hod = x.time.hour + x.time.minute / 60.0
        r["hod_s"], r["hod_c"] = float(np.sin(2 * np.pi * hod / 24)), float(np.cos(2 * np.pi * hod / 24))
        r["b_tot24"] = r["c_24h"]
        r["b_tot7"] = r["c_7d"]
        r["b_mag"] = r["t_mag"]
        hi = np.searchsorted(tm, x.tmin + 7 * 1440, side="right")
        fut = np.arange(i0 + 1, hi)
        y = 0
        if len(fut):
            cand = fut[mg[fut] >= 5.0]
            if len(cand):
                y = int((hav_km(x.Lat, x.Lon, la[cand], lo[cand]) <= 150).any())
        r["y"] = y
        rows.append(r)
    df = pd.DataFrame(rows)
    feats = [c for c in df.columns if c not in ("time", "y", "b_mag", "b_tot24", "b_tot7")]
    print(f"triggers={len(df)} feats={len(feats) + 3} (incl. baseline)")
    tr = df[df.time < "2020-01-01"].reset_index(drop=True)
    te = df[df.time >= "2020-01-01"].reset_index(drop=True)
    print(f"train={len(tr)} pos={tr.y.sum()} | test={len(te)} pos={te.y.sum()}")
    med = tr[feats].median().fillna(0.0)
    Xtr = tr[feats].fillna(med).to_numpy()
    Xte = te[feats].fillna(med).to_numpy()
    ytr, yte = tr.y.to_numpy(), te.y.to_numpy()
    nva = int(0.15 * len(tr))
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=50,
                                         l2_regularization=1.0, early_stopping=True,
                                         random_state=63)
    gbm.fit(Xtr[:-nva], ytr[:-nva])
    p_gbm = gbm.predict_proba(Xte)[:, 1]
    allf = ["b_mag", "b_tot24", "b_tot7", *feats]
    Atr = tr[allf].fillna(med).to_numpy()
    Ate = te[allf].fillna(med).to_numpy()
    sc = StandardScaler().fit(Atr)
    lg = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(Atr), ytr)
    scb = StandardScaler().fit(tr[["b_mag", "b_tot24", "b_tot7"]].to_numpy())
    lb = LogisticRegression(C=1.0).fit(scb.transform(tr[["b_mag", "b_tot24", "b_tot7"]].to_numpy()), ytr)
    p_log = lg.predict_proba(sc.transform(Ate))[:, 1]
    p_base = lb.predict_proba(scb.transform(te[["b_mag", "b_tot24", "b_tot7"]].to_numpy()))[:, 1]

    def ll(y, p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return y * np.log(p) + (1 - y) * np.log1p(-p)

    for name, p in [("GBM", p_gbm), ("logit60", p_log)]:
        ig = (ll(yte, p) - ll(yte, p_base)).sum() / yte.sum()
        print(f"IG {name} vs E1 = {ig:+.4f} | AUC={roc_auc_score(yte, p):.3f} (base "
              f"{roc_auc_score(yte, p_base):.3f}) PR={average_precision_score(yte, p):.3f} (base {yte.mean():.3f})")
        mo = te.time.dt.to_period("M").astype(str).to_numpy()
        um = np.unique(mo)
        dll = ll(yte, p) - ll(yte, p_base)
        boots = []
        for _ in range(2000):
            ms = RNG.choice(um, size=len(um), replace=True)
            m = np.isin(mo, ms)
            if yte[m].sum() > 0:
                boots.append(dll[m].sum() / yte[m].sum())
        boots = np.array(boots)
        lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
        cnt = 0
        for _ in range(5000):
            yp = RNG.permutation(yte)
            if yp.sum() == 0:
                continue
            cnt += ((ll(yp, p) - ll(yp, p_base)).sum() / yp.sum()) >= ig
        print(f"  IC95% [{lo:+.4f}, {hi:+.4f}] p={(1 + cnt) / 5001:.4f}")


if __name__ == "__main__":
    sys.exit(main())
