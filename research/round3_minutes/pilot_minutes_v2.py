"""Piloto v2: escalamiento a 60 min con microsismicidad Riedel (round3_minutes).

Triggers ML>=4 -> target: >=1 evento ML>=5 dentro de 60 min / 150 km.
Features incluyen conteo de microsismos ML>=2.5 en ultimos 10 min.
Comparacion primaria honesta: full vs mag-only (no vs clima descalibrado).
Vease PREREGISTRATION_V2.md. Uso: python3 research/round3_minutes/pilot_minutes_v2.py
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

CAT = "research/data/round2_enhanced_catalog/riedel/Primary_Catalogue.csv"
RNG = np.random.default_rng(1)
R_KM = 6371.0


def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def load():
    d = pd.read_csv(CAT, usecols=["Date", "Time", "Lat", "Lon", "Dep", "Ml"])
    d = d.dropna(subset=["Ml"]).reset_index(drop=True)  # 9253 eventos sin Ml se excluyen
    d["time"] = pd.to_datetime(d["Date"] + " " + d["Time"], format="mixed")
    d = d.sort_values("time").reset_index(drop=True)
    d["tmin"] = (d.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
    return d


def build(d):
    tm = d.tmin.to_numpy()
    la, lo, mg = (d[c].to_numpy() for c in ["Lat", "Lon", "Ml"])
    trig = d[(d.Ml >= 4.0) & (d.time >= "2017-01-01") & (d.time < "2021-11-15")].reset_index()
    rows = []
    for _, r in trig.iterrows():
        i0 = int(r["index"])
        # pasado 24h (searchsorted: rapido)
        lo24 = np.searchsorted(tm, r.tmin - 1440, side="left")
        past = np.arange(lo24, i0)
        dd = hav_km(r.Lat, r.Lon, la[past], lo[past]) if len(past) else np.array([])
        near = past[dd <= 150]
        dt = r.tmin - tm[near]
        m_near = mg[near]
        x1 = float(np.log1p(np.sum((dt <= 60) & (m_near >= 2.5))))
        x2 = float(r.Ml)
        dt4 = dt[m_near >= 4.0]
        x3 = float(np.log10(1 + dt4.min())) if len(dt4) else float(np.log10(1 + 1440))
        x4 = float(m_near.max()) if len(near) else 4.0
        x5 = float(np.log1p(np.sum((dt <= 10) & (m_near >= 2.5))))
        # target: futuro 60 min / 150 km, ML>=5
        hi = np.searchsorted(tm, r.tmin + 60, side="right")
        fut = np.arange(i0 + 1, hi)
        y = 0
        if len(fut):
            cand = fut[mg[fut] >= 5.0]
            if len(cand):
                y = int((hav_km(r.Lat, r.Lon, la[cand], lo[cand]) <= 150).any())
        rows.append({"time": r.time, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "y": y, "i0": i0})
    return pd.DataFrame(rows)


def ll_bern(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def main():
    print("cargando riedel...", flush=True)
    d = load()
    print(f"eventos: {len(d)}; construyendo triggers...", flush=True)
    df = build(d)
    tr = df[df.time < "2019-01-01"].reset_index(drop=True)
    te = df[df.time >= "2019-01-01"].reset_index(drop=True)
    print(f"train triggers={len(tr)} pos={tr.y.sum()} rate={tr.y.mean():.4f}")
    print(f"test  triggers={len(te)} pos={te.y.sum()} rate={te.y.mean():.4f}")
    # diagnostico replicas en test (preregistrado)
    tm = d.tmin.to_numpy(); la = d.Lat.to_numpy(); lo = d.Lon.to_numpy(); mg = d.Ml.to_numpy()
    trows = d[(d.Ml >= 4.0) & (d.time >= "2019-01-01") & (d.time < "2021-11-15")].reset_index(drop=True)
    n_after = 0; npos = int(te.y.sum())
    for _, r in te[te.y == 1].iterrows():
        i0 = int(r.i0)
        hi = np.searchsorted(tm, tm[i0] + 60, side="right")
        fut = np.arange(i0 + 1, hi)
        cand = fut[mg[fut] >= 5.0]
        dd = hav_km(la[i0], lo[i0], la[cand], lo[cand])
        tgt = cand[dd <= 150][0]
        pre = (tm < tm[tgt]) & (tm >= tm[tgt] - 30 * 1440)
        if mg[pre & (hav_km(la[tgt], lo[tgt], la, lo) <= 150)].max(initial=0) >= mg[tgt]:
            n_after += 1
    print(f"positivos test con mayor previo 30d/150km (replicas): {n_after}/{npos} ({n_after / max(npos, 1):.0%})")

    cols = ["x1", "x2", "x3", "x4", "x5"]
    sc = StandardScaler().fit(tr[cols].to_numpy())
    Xtr, Xte = sc.transform(tr[cols].to_numpy()), sc.transform(te[cols].to_numpy())
    ytr, yte = tr.y.to_numpy(), te.y.to_numpy()
    full = LogisticRegression(C=1.0).fit(Xtr, ytr)
    mago = LogisticRegression(C=1.0).fit(Xtr[:, [1]], ytr)
    nox5 = LogisticRegression(C=1.0).fit(Xtr[:, :4], ytr)
    p_full = full.predict_proba(Xte)[:, 1]
    p_mago = mago.predict_proba(Xte[:, [1]])[:, 1]
    p_nox5 = nox5.predict_proba(Xte[:, :4])[:, 1]
    p_cltr = np.full(len(te), ytr.mean())
    p_clte = np.full(len(te), yte.mean())
    npos = yte.sum()
    ig_fm = (ll_bern(yte, p_full) - ll_bern(yte, p_mago)).sum() / npos
    ig_fctr = (ll_bern(yte, p_full) - ll_bern(yte, p_cltr)).sum() / npos
    ig_fcte = (ll_bern(yte, p_full) - ll_bern(yte, p_clte)).sum() / npos
    ig_fnx5 = (ll_bern(yte, p_full) - ll_bern(yte, p_nox5)).sum() / npos
    print(f"\nIG full vs mag-only (PRIMARIA) = {ig_fm:+.4f} nats/evento")
    print(f"IG full vs sin-x5 (ablacion 10min) = {ig_fnx5:+.4f} nats/evento")
    print(f"IG full vs clima-train = {ig_fctr:+.4f} | vs clima-test(oraculo) = {ig_fcte:+.4f}")
    print(f"AUC full={roc_auc_score(yte, p_full):.3f} mago={roc_auc_score(yte, p_mago):.3f}")
    print(f"PR-AUC full={average_precision_score(yte, p_full):.3f} base={yte.mean():.3f}")
    print(f"coefs (std): {dict(zip(cols, np.round(full.coef_[0], 3)))}")

    years = te.time.dt.year.to_numpy(); uy = np.unique(years)
    dll = ll_bern(yte, p_full) - ll_bern(yte, p_mago)
    boots = []
    for _ in range(2000):
        ys = RNG.choice(uy, size=len(uy), replace=True)
        m = np.isin(years, ys)
        if yte[m].sum() > 0:
            boots.append(dll[m].sum() / yte[m].sum())
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    print(f"IC95% IG full-vs-mag-only: [{lo:+.4f}, {hi:+.4f}]")
    cnt = 0; B = 5000
    for _ in range(B):
        yp = RNG.permutation(yte)
        if yp.sum() == 0:
            continue
        cnt += ((ll_bern(yp, p_full) - ll_bern(yp, p_mago)).sum() / yp.sum()) >= ig_fm
    p = (1 + cnt) / (B + 1)
    print(f"p permutacion (aprox) = {p:.4f}")
    print("veredicto:", "NULO (piloto)" if (lo <= 0 or p >= 0.05) else "senal nominal (revisar replicas)")


if __name__ == "__main__":
    sys.exit(main())
