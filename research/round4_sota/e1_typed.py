"""E1: excitacion cruzada por tipo Riedel (round4_sota). Vease PREREG_E1_TYPED.md."""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

CAT = "research/data/round2_enhanced_catalog/riedel/Primary_Catalogue.csv"
CLEAN = "research/data/round2_enhanced_catalog/riedel/primary_clean.parquet"
RNG = np.random.default_rng(7)
R_KM = 6371.0
TYPES = ["Interplate", "Intraslab", "Outerrise", "Crustal"]


def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def load_clean():
    try:
        d = pd.read_parquet(CLEAN)
        print(f"catálogo limpio cacheado: {len(d)}")
        return d
    except Exception:
        pass
    r = pd.read_csv(CAT, usecols=["Date", "Time", "Lat", "Lon", "Dep", "Ml", "Type"])
    r["time"] = pd.to_datetime(r["Date"] + " " + r["Time"], format="mixed")
    r = r.dropna(subset=["Ml"]).drop_duplicates(subset=["time", "Lat", "Lon", "Ml"])
    r = r.sort_values("time").reset_index(drop=True)
    r["tmin"] = (r.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
    r.to_parquet(CLEAN, index=False)
    print(f"catálogo limpio guardado: {len(r)} (dedup+ordenado)")
    return r


def build(d):
    tm = d.tmin.to_numpy()
    la, lo, mg = (d[c].to_numpy() for c in ["Lat", "Lon", "Ml"])
    ty = d.Type.to_numpy()
    trig = d[(d.Ml >= 4.0) & (d.time >= "2017-03-01") & (d.time < "2021-11-15")].reset_index()
    rows = []
    for _, x in trig.iterrows():
        i0 = int(x["index"])
        lo30 = np.searchsorted(tm, x.tmin - 30 * 1440, side="left")
        pre = np.arange(lo30, i0)
        dd = hav_km(x.Lat, x.Lon, la[pre], lo[pre]) if len(pre) else np.array([])
        near30 = pre[dd <= 150]
        if len(near30) and mg[near30].max() >= x.Ml:
            continue  # no quiet
        lo7 = np.searchsorted(tm, x.tmin - 7 * 1440, side="left")
        w7 = np.arange(lo7, i0)
        dd7 = hav_km(x.Lat, x.Lon, la[w7], lo[w7]) if len(w7) else np.array([])
        n7 = w7[dd7 <= 150]
        dt7 = x.tmin - tm[n7]
        m7 = mg[n7]
        t7 = ty[n7]
        m25 = m7 >= 2.5
        row = {"time": x.time,
               "b_mag": float(x.Ml),
               "b_tot24": float(np.log1p(np.sum((dt7 <= 1440) & m25))),
               "b_tot7": float(np.log1p(m25.sum()))}
        for t in TYPES:
            row[f"t_{t}7"] = float(np.log1p(np.sum(m25 & (t7 == t))))
        hi = np.searchsorted(tm, x.tmin + 7 * 1440, side="right")
        fut = np.arange(i0 + 1, hi)
        y = 0
        if len(fut):
            cand = fut[mg[fut] >= 5.0]
            if len(cand):
                y = int((hav_km(x.Lat, x.Lon, la[cand], lo[cand]) <= 150).any())
        row["y"] = y
        rows.append(row)
    return pd.DataFrame(rows)


def ll_bern(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def main():
    d = load_clean()
    print("construyendo triggers quiet...", flush=True)
    df = build(d)
    tr = df[df.time < "2020-01-01"].reset_index(drop=True)
    te = df[df.time >= "2020-01-01"].reset_index(drop=True)
    print(f"train quiet={len(tr)} pos={tr.y.sum()} rate={tr.y.mean():.3f}")
    print(f"test  quiet={len(te)} pos={te.y.sum()} rate={te.y.mean():.3f}")
    base_cols = ["b_mag", "b_tot24", "b_tot7"]
    full_cols = base_cols + [f"t_{t}7" for t in TYPES]
    sc = StandardScaler().fit(tr[full_cols].to_numpy())
    Xtr, Xte = sc.transform(tr[full_cols].to_numpy()), sc.transform(te[full_cols].to_numpy())
    ytr, yte = tr.y.to_numpy(), te.y.to_numpy()
    full = LogisticRegression(C=1.0).fit(Xtr, ytr)
    base = LogisticRegression(C=1.0).fit(Xtr[:, :3], ytr)
    p_full = full.predict_proba(Xte)[:, 1]
    p_base = base.predict_proba(Xte[:, :3])[:, 1]
    ig = (ll_bern(yte, p_full) - ll_bern(yte, p_base)).sum() / yte.sum()
    print(f"\nIG full vs baseline = {ig:+.4f} nats/evento (PRIMARIA)")
    print(f"AUC full={roc_auc_score(yte, p_full):.3f} base={roc_auc_score(yte, p_base):.3f}")
    print(f"PR-AUC full={average_precision_score(yte, p_full):.3f} base={yte.mean():.3f}")
    print("coefs:", dict(zip(full_cols, np.round(full.coef_[0], 3))))
    print("signo Intraslab (Bouchon predice >0):",
          f"{full.coef_[0][full_cols.index('t_Intraslab7')]:+.3f}")
    mo = te.time.dt.to_period("M").astype(str).to_numpy()
    um = np.unique(mo)
    dll = ll_bern(yte, p_full) - ll_bern(yte, p_base)
    boots = []
    for _ in range(2000):
        ms = RNG.choice(um, size=len(um), replace=True)
        m = np.isin(mo, ms)
        if yte[m].sum() > 0:
            boots.append(dll[m].sum() / yte[m].sum())
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    print(f"IC95% mensual: [{lo:+.4f}, {hi:+.4f}]")
    cnt = 0
    for _ in range(5000):
        yp = RNG.permutation(yte)
        if yp.sum() == 0:
            continue
        cnt += ((ll_bern(yp, p_full) - ll_bern(yp, p_base)).sum() / yp.sum()) >= ig
    p = (1 + cnt) / 5001
    print(f"p permutacion = {p:.4f}")
    print("veredicto:", "NULO" if (lo <= 0 or p >= 0.05) else "SEÑAL (replicar)")


if __name__ == "__main__":
    sys.exit(main())
