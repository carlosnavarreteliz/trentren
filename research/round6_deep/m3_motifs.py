"""M3: motivos n-grama antes de ML>=5.5 vs controles. Vease PREREG_M3_MOTIFS.md."""
import sys
import numpy as np
import pandas as pd
from collections import Counter

CLEAN = "research/data/round2_enhanced_catalog/riedel/primary_clean.parquet"
RNG = np.random.default_rng(63)
R_KM = 6371.0


def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def sym(m, z, t):
    mc = "A" if m < 3.5 else ("B" if m < 4.5 else "C")
    dc = "S" if z <= 60 else "D"
    tc = {"Interplate": "I", "Intraslab": "N"}.get(t, "X")
    return mc + dc + tc


def main():
    d = pd.read_parquet(CLEAN)
    d["time"] = pd.to_datetime(d["time"])
    d["tmin"] = (d.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
    d = d.sort_values("time").reset_index(drop=True)
    tm, la, lo, mg = (d[c].to_numpy() for c in ["tmin", "Lat", "Lon", "Ml"])
    d["s"] = [sym(m, z, t) for m, z, t in zip(d.Ml, d.Dep, d.Type)]

    def seq_before(tmin_c, lat, lon, k=20):
        hi = np.searchsorted(tm, tmin_c, side="left")
        pre = np.arange(max(0, hi - 20000), hi)
        dd = hav_km(lat, lon, la[pre], lo[pre])
        near = pre[(dd <= 150) & (mg[pre] >= 2.5)]
        return tuple(d.s.to_numpy()[near[-k:]]) if len(near) else ()

    big = d[d.Ml >= 5.5].reset_index()
    cases = []
    for _, x in big.iterrows():
        i0 = int(x["index"])
        w = (np.abs(tm - x.tmin) <= 30 * 1440)
        if mg[w & (hav_km(x.Lat, x.Lon, la, lo) <= 150)].max(initial=0) > x.Ml:
            continue
        cases.append(x)
    print(f"mainshock-proxy ML>=5.5: {len(cases)}")
    rows = []
    for x in cases:
        s = seq_before(x.tmin, x.Lat, x.Lon)
        if len(s) >= 8:
            rows.append(("case", x.time.year, s))
        for _ in range(5):
            for _try in range(20):
                dt_days = RNG.integers(-365, 365)
                t_c = x.time + pd.Timedelta(days=int(dt_days))
                if t_c <= d.time.min() + pd.Timedelta(days=60) or t_c >= d.time.max() - pd.Timedelta(days=30):
                    continue
                w = (np.abs(tm - (t_c - pd.Timestamp("1970-01-01")).total_seconds() / 60.0) <= 60 * 1440)
                if mg[w & (hav_km(x.Lat, x.Lon, la, lo) <= 150)].max(initial=0) >= 5.5:
                    continue
                s = seq_before((t_c - pd.Timestamp("1970-01-01")).total_seconds() / 60.0, x.Lat, x.Lon)
                if len(s) >= 8:
                    rows.append(("ctrl", x.time.year, s))
                    break
    print(f"secuencias: casos={sum(1 for r in rows if r[0]=='case')} controles={sum(1 for r in rows if r[0]=='ctrl')}")

    def ngrams(s, n):
        return {" ".join(s[i:i + n]) for i in range(len(s) - n + 1)}

    disc = [r for r in rows if r[1] <= 2019]
    val = [r for r in rows if r[1] >= 2020]
    print(f"descubrimiento: casos={sum(1 for r in disc if r[0]=='case')} | validacion: casos={sum(1 for r in val if r[0]=='case')}")
    cnt_case, cnt_ctrl = Counter(), Counter()
    nc = sum(1 for r in disc if r[0] == "case")
    nk = sum(1 for r in disc if r[0] == "ctrl")
    for lab, _, s in disc:
        for n in (2, 3, 4):
            for g in ngrams(s, n):
                (cnt_case if lab == "case" else cnt_ctrl)[g] += 1
    cands = {g: (cnt_case[g] / nc, cnt_ctrl[g] / nk) for g in cnt_case
             if cnt_case[g] >= 3 and cnt_case[g] / nc > 1.8 * max(cnt_ctrl[g] / nk, 1e-9)}
    print(f"motivos candidatos (suporte>=3, x1.8): {len(cands)}")
    if not cands:
        print("veredicto: NULO (ni candidatos)")
        return
    # permutacion en descubrimiento
    seqs = [(lab == "case", s) for lab, _, s in disc]
    sig = {}
    for g in cands:
        hits = [1 if lab else 0 for lab, _ in seqs]
        pres = [1 if any(g == " ".join(s[i:i + len(g.split())]) for i in range(len(s) - len(g.split()) + 1)) else 0
                for _, s in seqs]
        c = 0
        for _ in range(1000):
            ph = RNG.permutation(hits)
            a = [pres[i] for i in range(len(pres)) if ph[i] == 1]
            b = [pres[i] for i in range(len(pres)) if ph[i] == 0]
            if (np.mean(a) - np.mean(b)) >= (np.mean([pres[i] for i in range(len(pres)) if hits[i] == 1]) -
                                             np.mean([pres[i] for i in range(len(pres)) if hits[i] == 0])):
                c += 1
        p = (1 + c) / 1001
        if p < 0.05 / max(len(cands), 1):
            sig[g] = p
    print(f"motivos significativos (Bonferroni): {len(sig)}")
    for g, p in sig.items():
        print(f"  {g} p={p:.4f} disc=({cands[g][0]:.2f} vs {cands[g][1]:.2f})")
    if not sig:
        print("veredicto: NULO")
        return
    # validacion temporal
    for g in sig:
        n = len(g.split())
        vc = [r for r in val if r[0] == "case"]
        vk = [r for r in val if r[0] == "ctrl"]
        def rate(rs):
            if not rs:
                return 0.0
            return float(np.mean([any(g == " ".join(s[i:i + n]) for i in range(len(s) - n + 1)) for _, _, s in rs]))
        print(f"validacion {g}: casos2020+={rate(vc):.2f} controles={rate(vk):.2f}")
    print("ver reporte completo en M3_RESULT (revisar rates de validacion)")


if __name__ == "__main__":
    sys.exit(main())
