"""Verificacion de catalogos (round4_sota): duplicados, orden, Mc, tipos de magnitud."""
import numpy as np
import pandas as pd

print("=" * 70, "\n1) USGS Chile (data/raw/chile_usgs_1900_2026.csv)")
u = pd.read_csv("data/raw/chile_usgs_1900_2026.csv", low_memory=False)
print(f"filas={len(u)}")
u["time"] = pd.to_datetime(u["time"], format="ISO8601", utc=True).dt.tz_convert(None)
print("ordenado por tiempo:", bool((u.time.diff().dropna() >= pd.Timedelta(0)).all()))
print("ids duplicados:", int(u.duplicated("id").sum()))
print("tipos:", u["type"].value_counts().to_dict())
print("magType:", u["magType"].value_counts().to_dict())
print("mag NaN:", int(u.mag.isna().sum()), "| mag<0:", int((u.mag < 0).sum()))
print("coords NaN:", int(u.latitude.isna().sum() + u.longitude.isna().sum()))
print("depth NaN:", int(u.depth.isna().sum()), "| depth<0:", int((u.depth < 0).sum()))
q = u[(u.type == "earthquake") & u.latitude.between(-46, -17) & u.longitude.between(-78, -66)]
q = q.dropna(subset=["mag"]).sort_values("time").reset_index(drop=True)
print(f"en caja harness: {len(q)}")
print("duplicados exactos (t,lat,lon,mag):",
      int(q.duplicated(subset=["time", "latitude", "longitude", "mag"]).sum()))
same_min = q[q.time.diff().dt.total_seconds().fillna(60) < 1]
print(f"pares <1s aparte: {len(same_min)} (posibles duplicados de red)")
# Mc por decada (max curvature + 0.2)
for a, b in [(1980, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2026)]:
    m = q[q.time.dt.year.between(a, b)].mag.to_numpy()
    v, c = np.unique(np.round(m / 0.1) * 0.1, return_counts=True)
    mc = v[np.argmax(c)] + 0.2
    print(f"  Mc {a}-{b}: {mc:.1f} (n={len(m)}, n>=4.7={(m >= 4.7).sum()})")

print("=" * 70, "\n2) Riedel primario")
r = pd.read_csv("research/data/round2_enhanced_catalog/riedel/Primary_Catalogue.csv",
                usecols=["Date", "Time", "Lat", "Lon", "Dep", "Ml", "Type"])
print(f"filas={len(r)} Ml NaN={int(r.Ml.isna().sum())}")
r["time"] = pd.to_datetime(r["Date"] + " " + r["Time"], format="mixed")
print("rango:", r.time.min(), "->", r.time.max(), "| ordenado:",
      bool((r.time.diff().dropna() >= pd.Timedelta(0)).all()))
print("tipos:", r["Type"].value_counts().to_dict())
print("duplicados exactos:", int(r.duplicated(subset=["time", "Lat", "Lon", "Ml"]).sum()))
print("lats fuera 18-42S:", int((~r.Lat.between(-42, -18)).sum()))

print("=" * 70, "\n3) IPOC")
c = ["year", "month", "day", "hour", "minute", "sec", "lat", "lon", "dep",
     "ML", "MLsd", "MA", "MAsd", "id"]
d = pd.read_csv("research/data/round2_enhanced_catalog/ipoc/2023-004_Sippl-et-al_IPOC2022-earthquake-catalogue.txt",
                sep=r"\s+", comment="#", names=c)
print(f"filas={len(d)} ML NaN={int(d.ML.isna().sum())}")
d["time"] = pd.to_datetime(d[["year", "month", "day", "hour", "minute"]]) + pd.to_timedelta(d["sec"], unit="s")
print("rango:", d.time.min(), "->", d.time.max(), "| ordenado:",
      bool((d.time.diff().dropna() >= pd.Timedelta(0)).all()))
print("duplicados exactos:", int(d.duplicated(subset=["time", "lat", "lon", "ML"]).sum()))
print("ML vs MA (media diff):", float((d.ML - d.MA).mean()))
print("=" * 70)
