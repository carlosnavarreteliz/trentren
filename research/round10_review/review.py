"""Round10: (A) prueba de truncacion causal, (B) auditoria anos anomalos,
(C) patrones directos: tasas base de escalamiento y foreshocks Riedel."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import forecast_short as F

print("=" * 72)
print("A) MEMORIZACION/FUGA: features con catalogo truncado en el trigger")
d = F.load_local()
trig = d[(d.mag >= 4.7) & (d.time >= "2024-01-01")].head(300).reset_index()
tm, la, lo, mg, dp = (d[c].to_numpy() for c in ["tmin", "latitude", "longitude", "mag", "depth"])
mx = 0.0
for _, r in trig.iterrows():
    i0 = int(r["index"])
    f_full = F.trigger_features(r, i0, tm, la, lo, mg, dp)
    cut = d[d.time < r.time]
    tm2 = cut.tmin.to_numpy()
    la2, lo2, mg2, dp2 = (cut[c].to_numpy() for c in ["latitude", "longitude", "mag", "depth"])
    f_cut = F.trigger_features(r, len(cut) - 1 + 1, tm2, la2, lo2, mg2, dp2) \
        if len(cut) else {k: 0.0 for k in f_full}
    # trigger_features usa indices < i0: con catalogo cortado, i0' = len(cut)
    f_cut = F.trigger_features(r, len(cut), tm2, la2, lo2, mg2, dp2)
    for k in f_full:
        mx = max(mx, abs(f_full[k] - f_cut[k]))
print(f"max |feature(completo) - feature(truncado)| en 300 triggers = {mx:.2e}")
print("features del modelo:", F.COLS_V2, "-> sin lat/lon crudas: no puede memorizar lugares")
print("splits cronologicos train<=2020 / calib 2021-23 / test 2024+; test congelado benchmark;")
print("log prospectivo firmado por issued_at desde 2026-10-28.")

print("=" * 72)
print("B) ANOS ANOMALOS: tasas y composicion USGS por ano (caja harness)")
u = d.copy()
u["yr"] = u.time.dt.year
g = u[u.yr.between(2010, 2026)].groupby("yr").agg(
    n47=("mag", lambda s: int((s >= 4.7).sum())),
    n5=("mag", lambda s: int((s >= 5.0).sum())),
    n6=("mag", lambda s: int((s >= 6.0).sum())),
    med_dep=("depth", "median"))
print(g.to_string())
raw = pd.read_csv("data/raw/chile_usgs_1900_2026.csv", usecols=["time", "magType", "type", "time"])
raw = raw[raw["type"] == "earthquake"]
raw["yr"] = pd.to_datetime(raw["time"], format="ISO8601", utc=True).dt.year
print("\n% mb por ano (era post-expansion CSN):")
for y in (2011, 2014, 2018, 2022, 2025):
    s = raw[raw.yr == y]
    print(f"  {y}: mb={(s.magType == 'mb').mean():.0%} mww={(s.magType == 'mww').mean():.0%} total={len(s)}")

print("=" * 72)
print("C1) TECHO EMPIRICO: P(escala en 24h) segun magnitud del trigger (USGS 1990-2026)")
dd = F.build_dataset(d)
for elo, ehi in [(4.7, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 9.9)]:
    s = dd[(dd.x2 >= elo) & (dd.x2 < ehi)]
    print(f"  trigger M[{elo},{ehi}): n={len(s)} P60m={s.y_60.mean():.3f} P24h={s.y_1440.mean():.3f}")

print("C2) FORESHOCKS Riedel: M>=5.5 con precursor ML>=4 en 7d/50km vs azar")
r = pd.read_parquet("research/data/round2_enhanced_catalog/riedel/primary_clean.parquet")
r["time"] = pd.to_datetime(r["time"])
r["tmin"] = (r.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
tmr, lar, lor, mgr = (r[c].to_numpy() for c in ["tmin", "Lat", "Lon", "Ml"])
big = r[r.Ml >= 5.5].reset_index()
rng = np.random.default_rng(3)
hits_c = hits_r = 0
n = 0
for _, x in big.iterrows():
    def has_pre(t0):
        j = (tmr < t0) & (tmr >= t0 - 7 * 1440) & (mgr >= 4.0)
        if not j.sum():
            return 0
        idx = np.where(j)[0]
        return int((F.hav_km(x.Lat, x.Lon, lar[idx], lor[idx]) <= 50).any())
    hits_c += has_pre(x.tmin)
    hits_r += has_pre(x.tmin + rng.integers(30, 300) * 1440 * rng.choice([-1, 1]))
    n += 1
print(f"  casos M>=5.5: {n} | con foreshock 7d/50km: {hits_c} ({hits_c/n:.0%}) vs control desplazado: {hits_r} ({hits_r/n:.0%})")
