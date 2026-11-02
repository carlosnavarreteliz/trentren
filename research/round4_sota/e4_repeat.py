"""E4: familias repeaters via S-P + test de escalamiento. Vease PREREG_E4_REPEAT.md.

Paso 1 (pesado, cacheable): pares intra-bin RMS(S-P diff)<0.5s, >=6 estaciones
comunes P+S -> familias (componentes conexas >=3). Guarda families.parquet.
Paso 2: test E1-style con feature causal de actividad de familias.
Uso: python3 research/round4_sota/e4_repeat.py [--families-only]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

CLEAN = "research/data/round2_enhanced_catalog/riedel/primary_clean.parquet"
PICKS = "research/data/round2_enhanced_catalog/riedel/Riedel-Hornig_et_al_Primary_Picks.csv"
FAM = "research/round4_sota/e4_families.parquet"
RNG = np.random.default_rng(31)
R_KM = 6371.0
BINS = [-33, -32, -31]


def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def build_families():
    d = pd.read_parquet(CLEAN, columns=["time", "Lat", "Lon", "Ml", "Type"])
    d["time"] = pd.to_datetime(d["time"])
    d["bin"] = np.floor(d.Lat).astype(int)
    scope = d[(d.Type == "Interplate") & (d.Ml >= 2.5) & (d.bin.isin(BINS))].copy()
    print(f"eventos en alcance: {len(scope)}", flush=True)
    # S-P por evento-estacion (solo eventos en alcance; Event_idx Riedel = id orig)
    # NOTE: primary_clean no guarda id; re-mapeamos por (time,Lat,Lon,Ml)
    r0 = pd.read_csv("research/data/round2_enhanced_catalog/riedel/Primary_Catalogue.csv",
                     usecols=["id", "Date", "Time", "Lat", "Lon", "Ml"])
    r0["time"] = pd.to_datetime(r0["Date"] + " " + r0["Time"], format="mixed")
    key = lambda df, t, la, lo, m: (t.astype(str) + "|" + la.round(5).astype(str) + "|" +
                                    lo.round(5).astype(str) + "|" + m.round(3).astype(str))
    scope["_k"] = key(scope, scope.time, scope.Lat, scope.Lon, scope.Ml)
    r0["_k"] = key(r0, r0.time, r0.Lat, r0.Lon, r0.Ml)
    idmap = dict(zip(r0._k, r0.id))
    scope["eid"] = scope._k.map(idmap)
    print(f"con id Riedel: {int(scope.eid.notna().sum())}", flush=True)
    want = set(scope.eid.dropna().astype(int).tolist())
    sp = {}  # eid -> {station: SP}
    for ch in pd.read_csv(PICKS, chunksize=2_000_000):
        ch = ch[ch.Event_idx.isin(want)]
        if not len(ch):
            continue
        ch = ch.drop_duplicates(subset=["Event_idx", "Station", "Type"])
        ch["DateTime"] = pd.to_datetime(ch["DateTime"], format="mixed", utc=True)
        pv = ch[ch.Type == "P"].set_index(["Event_idx", "Station"])["DateTime"]
        sv = ch[ch.Type == "S"].set_index(["Event_idx", "Station"])["DateTime"]
        both = pd.concat([pv, sv], axis=1, keys=["P", "S"]).dropna()
        both["SP"] = (both["S"] - both["P"]).dt.total_seconds()
        for (eid, sta), row in both["SP"].items():
            sp.setdefault(int(eid), {})[sta] = float(row)
    print(f"eventos con S-P: {len(sp)}", flush=True)
    eids = sorted(sp)
    pos = {e: i for i, e in enumerate(eids)}
    parent = list(range(len(eids)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    npairs, nsim = 0, 0
    eid_bin = dict(zip(scope.eid.dropna().astype(int), scope.bin))
    bybin = {}
    for e in eids:
        bybin.setdefault(eid_bin.get(e, -99), []).append(e)
    for b, elist in bybin.items():
        S = [sp[e] for e in elist]
        for i in range(len(elist)):
            si = S[i]
            for j in range(i + 1, len(elist)):
                sj = S[j]
                common = set(si) & set(sj)
                if len(common) < 6:
                    continue
                npairs += 1
                v = np.array([si[s] - sj[s] for s in common])
                if np.sqrt(np.mean(v ** 2)) < 0.5:
                    nsim += 1
                    union(pos[elist[i]], pos[elist[j]])
    print(f"pares evaluados={npairs} similares={nsim}", flush=True)
    fam = {}
    for e in eids:
        fam.setdefault(find(pos[e]), []).append(e)
    fams = [v for v in fam.values() if len(v) >= 3]
    print(f"familias(>=3): {len(fams)} eventos en familias: {sum(map(len, fams))}", flush=True)
    out = pd.DataFrame([(e, f) for f, v in enumerate(fams) for e in v],
                       columns=["eid", "fam"])
    out.to_parquet(FAM, index=False)
    return out


def ll_bern(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def run_test():
    fams = pd.read_parquet(FAM)
    famset = set(fams.eid.tolist())
    d = pd.read_parquet(CLEAN)
    d["time"] = pd.to_datetime(d["time"])
    # mapa eid->pos para causalidad (re-mapeo liviano)
    r0 = pd.read_csv("research/data/round2_enhanced_catalog/riedel/Primary_Catalogue.csv",
                     usecols=["id", "Date", "Time", "Lat", "Lon", "Ml"])
    r0["time"] = pd.to_datetime(r0["Date"] + " " + r0["Time"], format="mixed")
    r0 = r0.dropna(subset=["Ml"]).drop_duplicates(subset=["time", "Lat", "Lon", "Ml"])
    r0 = r0.sort_values("time").reset_index(drop=True)
    r0["tmin"] = (r0.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
    tm = r0.tmin.to_numpy()
    la, lo, mg = (r0[c].to_numpy() for c in ["Lat", "Lon", "Ml"])
    iseid = r0.id.to_numpy()
    # por trigger: eventos pasados 150km/30d que estan en familias con >=2 miembros previos
    famby = {}
    for eid, f in zip(fams.eid, fams.fam):
        famby.setdefault(f, []).append(eid)
    fam_times = {f: sorted(tm[np.isin(iseid, v)]) for f, v in famby.items()}
    trig = r0[(r0.Ml >= 4.0) & (r0.time >= "2017-03-01") & (r0.time < "2021-11-15")].reset_index()
    rows = []
    for _, x in trig.iterrows():
        i0 = int(x["index"])
        lo30 = np.searchsorted(tm, x.tmin - 30 * 1440, side="left")
        pre = np.arange(lo30, i0)
        dd = hav_km(x.Lat, x.Lon, la[pre], lo[pre]) if len(pre) else np.array([])
        near30 = pre[dd <= 150]
        if len(near30) and mg[near30].max() >= x.Ml:
            continue
        lo7 = np.searchsorted(tm, x.tmin - 7 * 1440, side="left")
        w7 = np.arange(lo7, i0)
        dd7 = hav_km(x.Lat, x.Lon, la[w7], lo[w7]) if len(w7) else np.array([])
        n7 = w7[dd7 <= 150]
        dt7 = x.tmin - tm[n7]
        m25 = mg[n7] >= 2.5
        # feature familias: miembros previos en 150km/30d cuyas familias tienen >=3 total
        # y >=2 miembros con t<trigger (causal)
        fcount = 0
        pe = iseid[near30]
        for f, tt in fam_times.items():
            nprev = sum(1 for t in tt if t < x.tmin)
            if nprev >= 2:
                fcount += sum(1 for e in famby[f]
                              if e in set(pe.tolist()) and tm[iseid == e][0] < x.tmin)
        row = {"time": x.time, "b_mag": float(x.Ml),
               "b_tot24": float(np.log1p(np.sum((dt7 <= 1440) & m25))),
               "b_tot7": float(np.log1p(m25.sum())),
               "x_fam30": float(np.log1p(fcount))}
        hi = np.searchsorted(tm, x.tmin + 7 * 1440, side="right")
        fut = np.arange(i0 + 1, hi)
        y = 0
        if len(fut):
            cand = fut[mg[fut] >= 5.0]
            if len(cand):
                y = int((hav_km(x.Lat, x.Lon, la[cand], lo[cand]) <= 150).any())
        row["y"] = y
        rows.append(row)
    df = pd.DataFrame(rows)
    tr = df[df.time < "2020-01-01"].reset_index(drop=True)
    te = df[df.time >= "2020-01-01"].reset_index(drop=True)
    print(f"train quiet={len(tr)} pos={tr.y.sum()} | test quiet={len(te)} pos={te.y.sum()}")
    cols = ["b_mag", "b_tot24", "b_tot7", "x_fam30"]
    sc = StandardScaler().fit(tr[cols].to_numpy())
    Xtr, Xte = sc.transform(tr[cols].to_numpy()), sc.transform(te[cols].to_numpy())
    ytr, yte = tr.y.to_numpy(), te.y.to_numpy()
    full = LogisticRegression(C=1.0).fit(Xtr, ytr)
    base = LogisticRegression(C=1.0).fit(Xtr[:, :3], ytr)
    p_full = full.predict_proba(Xte)[:, 1]
    p_base = base.predict_proba(Xte[:, :3])[:, 1]
    ig = (ll_bern(yte, p_full) - ll_bern(yte, p_base)).sum() / yte.sum()
    print(f"IG full vs baseline = {ig:+.4f} (PRIMARIA)")
    print(f"AUC full={roc_auc_score(yte, p_full):.3f} base={roc_auc_score(yte, p_base):.3f}")
    print(f"PR-AUC full={average_precision_score(yte, p_full):.3f} base={yte.mean():.3f}")
    print("coef x_fam30:", f"{full.coef_[0][3]:+.3f}")
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
    if "--families-only" in sys.argv:
        build_families()
    else:
        try:
            pd.read_parquet(FAM)
            print("familias cacheadas, corriendo test...")
        except Exception:
            build_families()
        run_test()
