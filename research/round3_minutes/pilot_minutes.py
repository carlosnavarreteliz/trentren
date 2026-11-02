"""Piloto chico: escalamiento a minutos (round3_minutes).

Trigger M>=4.7 -> target: >=1 evento M>=5.0 dentro de 60 min y 150 km.
Solo catalogo USGS local. Vease PREREGISTRATION.md.
Uso: python3 research/round3_minutes/pilot_minutes.py
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = "."
CAT = f"{ROOT}/data/raw/chile_usgs_1900_2026.csv"
RNG = np.random.default_rng(0)
R_KM = 6371.0


def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def load():
    d = pd.read_csv(CAT, usecols=["time", "latitude", "longitude", "mag", "type"])
    d = d[d["type"] == "earthquake"]
    d["time"] = pd.to_datetime(d["time"], format="ISO8601", utc=True).dt.tz_convert(None)
    d = d[d.latitude.between(-46, -17) & d.longitude.between(-78, -66)]
    d = d.dropna(subset=["mag"]).sort_values("time").reset_index(drop=True)
    d["tmin"] = (d.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
    return d


def build(d):
    tm, la, lo, mg = (d[c].to_numpy() for c in ["tmin", "latitude", "longitude", "mag"])
    trig = d[(d.mag >= 4.7) & (d.time >= "1990-01-01") & (d.time < "2026-10-01")].reset_index()
    rows = []
    for _, r in trig.iterrows():
        i0 = int(r["index"])  # posicion en d; pasado estricto = indices < i0
        past = np.arange(i0)
        dist_p = hav_km(r.latitude, r.longitude, la[past], lo[past]) if i0 else np.array([])
        near = past[dist_p <= 150] if i0 else np.array([], dtype=int)
        near47 = near[mg[near] >= 4.7]
        dt_min = r.tmin - tm[near47]  # > 0 siempre (pasado estricto)
        x1 = float(np.log1p(np.sum(dt_min <= 60)))
        x2 = float(r.mag)
        x3 = float(np.log10(1 + dt_min.min())) if len(dt_min) else float(np.log10(1 + 525600))
        x4 = float(mg[near[ (r.tmin - tm[near] <= 1440) ]].max()) if len(near) and np.any((r.tmin - tm[near] <= 1440)) else 4.2
        # target: futuro 60 min / 150 km, M>=5.0
        fut = (tm > r.tmin) & (tm <= r.tmin + 60) & (mg >= 5.0)
        y = 0
        if fut.sum():
            idx = np.where(fut)[0]
            y = int((hav_km(r.latitude, r.longitude, la[idx], lo[idx]) <= 150).any())
        rows.append({"time": r.time, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})
    return pd.DataFrame(rows)


def ll_bern(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def main():
    print("cargando catalogo...", flush=True)
    d = load()
    print("construyendo triggers/features (puede tardar ~1-2 min)...", flush=True)
    df = build(d)
    tr = df[df.time < "2018-01-01"].reset_index(drop=True)
    te = df[df.time >= "2018-01-01"].reset_index(drop=True)
    print(f"train triggers={len(tr)} pos={tr.y.sum()} rate={tr.y.mean():.4f}")
    print(f"test  triggers={len(te)} pos={te.y.sum()} rate={te.y.mean():.4f}")

    cols = ["x1", "x2", "x3", "x4"]
    sc = StandardScaler().fit(tr[cols].to_numpy())
    Xtr, Xte = sc.transform(tr[cols].to_numpy()), sc.transform(te[cols].to_numpy())
    ytr, yte = tr.y.to_numpy(), te.y.to_numpy()

    full = LogisticRegression(C=1.0).fit(Xtr, ytr)
    mago = LogisticRegression(C=1.0).fit(Xtr[:, [1]], ytr)
    p_full = full.predict_proba(Xte)[:, 1]
    p_mago = mago.predict_proba(Xte[:, [1]])[:, 1]
    p_clim = np.full(len(te), ytr.mean())

    npos = yte.sum()
    ig_full = (ll_bern(yte, p_full) - ll_bern(yte, p_clim)).sum() / npos
    ig_mago = (ll_bern(yte, p_mago) - ll_bern(yte, p_clim)).sum() / npos
    ig_fm = (ll_bern(yte, p_full) - ll_bern(yte, p_mago)).sum() / npos
    print(f"\nIG full vs clim  = {ig_full:+.4f} nats/evento")
    print(f"IG mago vs clim  = {ig_mago:+.4f} nats/evento")
    print(f"IG full vs mago  = {ig_fm:+.4f} nats/evento")
    print(f"AUC full={roc_auc_score(yte, p_full):.3f} mago={roc_auc_score(yte, p_mago):.3f}")
    print(f"PR-AUC full={average_precision_score(yte, p_full):.3f} base={yte.mean():.3f}")
    print(f"coefs (std): {dict(zip(cols, np.round(full.coef_[0], 3)))}")

    # bootstrap por ano para IG full vs clim
    years = te.time.dt.year.to_numpy()
    uy = np.unique(years)
    dll = ll_bern(yte, p_full) - ll_bern(yte, p_clim)
    boots = []
    for _ in range(2000):
        ys = RNG.choice(uy, size=len(uy), replace=True)
        m = np.isin(years, ys)
        if yte[m].sum() > 0:
            boots.append(dll[m].sum() / yte[m].sum())
    boots = np.array(boots)
    print(f"IC95% IG full-vs-clim: [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]")

    # permutacion de etiquetas en test (aprox): IG de predicciones fijas vs y mezclado
    cnt = 0
    B = 5000
    for _ in range(B):
        yp = RNG.permutation(yte)
        if yp.sum() == 0:
            continue
        igp = (ll_bern(yp, p_full) - ll_bern(yp, p_clim)).sum() / yp.sum()
        cnt += igp >= ig_full
    print(f"p permutacion (aprox) = {(1 + cnt) / (B + 1):.4f}")
    print("\nveredicto:", "NULO (piloto)" if (np.percentile(boots, 2.5) <= 0) else "senal nominal")


if __name__ == "__main__":
    sys.exit(main())
