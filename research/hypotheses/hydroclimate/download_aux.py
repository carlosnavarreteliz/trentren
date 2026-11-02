"""Elevation of CPC 0.5deg cells (OpenTopoData ETOPO1, fallback when Open-Meteo rate-limited) + ONI + PDO.
Usage: python -I download_aux.py <data_dir>"""
import sys, time
from pathlib import Path
import numpy as np, requests
OUT = Path(sys.argv[1])
ef = OUT / "elevation.npz"
if not ef.exists():
    lat = 89.75 - 0.5 * np.arange(213, 274); lon = 0.25 + 0.5 * np.arange(566, 589) - 360
    LA, LO = np.meshgrid(lat, lon, indexing="ij"); la, lo = LA.ravel(), LO.ravel(); el = []
    for i in range(0, len(la), 100):
        locs = "|".join(f"{a:.2f},{b:.2f}" for a, b in zip(la[i:i+100], lo[i:i+100]))
        for att in range(5):
            r = requests.get("https://api.opentopodata.org/v1/etopo1", params={"locations": locs}, timeout=60)
            if r.status_code == 200:
                break
            time.sleep(3 * (att + 1))
        r.raise_for_status()
        el += [x["elevation"] for x in r.json()["results"]]; time.sleep(1.2)
    np.savez_compressed(ef, elev=np.array(el, float).reshape(LA.shape), lat=lat, lon=lon, source="opentopodata etopo1")
    print("elevation ok")
for name, u in {"oni.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
                "pdo.txt": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"}.items():
    if not (OUT / name).exists():
        r = requests.get(u, timeout=60); r.raise_for_status(); (OUT / name).write_bytes(r.content); print(name, "ok")
