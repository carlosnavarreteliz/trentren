"""Download CPC daily precip/tmax/tmin (0.5 deg) subset over Chile via NOAA PSL OPeNDAP (.dods binary),
Open-Meteo elevations for the grid cells, NOAA ONI and PDO. Caches under the data dir given as argv[1].
Usage: python -I download.py /path/to/research/data/hydroclimate [vars=precip,tmax,tmin]"""
import sys, time, struct, json
from pathlib import Path
import numpy as np, requests

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
VARS = (sys.argv[2] if len(sys.argv) > 2 else "precip,tmax,tmin").split(",")
LAT0, LAT1 = 213, 273      # lat = 89.75-0.5*i -> -16.75 .. -46.25
LON0, LON1 = 566, 588      # lon = 0.25+0.5*j -> 283.25(-76.75) .. 293.75(-66.25)
URL = {"precip": "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_global_precip/precip.{y}.nc",
       "tmax": "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_global_temp/tmax.{y}.nc",
       "tmin": "https://psl.noaa.gov/thredds/dodsC/Datasets/cpc_global_temp/tmin.{y}.nc"}

def parse_dods(raw):
    k = raw.index(b"\nData:\n") + 7
    buf = raw[k:]; out = []; pos = 0
    # Grid: ARRAY float32, then maps time(float64), lat(float32), lon(float32)
    for dt in (">f4", ">f8", ">f4", ">f4"):
        n, n2 = struct.unpack(">ii", buf[pos:pos + 8]); pos += 8
        sz = np.dtype(dt).itemsize
        out.append(np.frombuffer(buf[pos:pos + n * sz], dtype=dt).astype(float)); pos += n * sz
    return out

def get(var, y, ndays):
    f = OUT / f"{var}_{y}.npz"
    if f.exists():
        return
    u = URL[var].format(y=y) + f".dods?{var}[0:1:{ndays-1}][{LAT0}:1:{LAT1}][{LON0}:1:{LON1}]"
    for att in range(5):
        try:
            r = requests.get(u, timeout=180); r.raise_for_status()
            a, t, lat, lon = parse_dods(r.content)
            a = a.reshape(len(t), len(lat), len(lon))
            a[(a < -1e3) | (a > 1e4)] = np.nan
            np.savez_compressed(f, data=a.astype(np.float32), time=t, lat=lat, lon=lon)
            print(var, y, a.shape, np.nanmean(a).round(3), flush=True); return
        except Exception as e:
            print("retry", var, y, e, flush=True); time.sleep(5 * (att + 1))
    print("FAILED", var, y)

def ndays_avail(var, y):
    r = requests.get(URL[var].format(y=y) + ".dds", timeout=60); r.raise_for_status()
    import re
    return int(re.search(r"time = (\d+)\]", r.text).group(1))

for var in VARS:
    for y in range(1979, 2027):
        if (OUT / f"{var}_{y}.npz").exists():
            continue
        nd = ndays_avail(var, y)
        get(var, y, nd); time.sleep(0.5)

# elevation of grid cells (Open-Meteo, up to 100 coords per call)
ef = OUT / "elevation.npz"
if not ef.exists():
    lat = 89.75 - 0.5 * np.arange(LAT0, LAT1 + 1); lon = 0.25 + 0.5 * np.arange(LON0, LON1 + 1) - 360
    LA, LO = np.meshgrid(lat, lon, indexing="ij"); la, lo = LA.ravel(), LO.ravel(); el = []
    for i in range(0, len(la), 100):
        r = requests.get("https://api.open-meteo.com/v1/elevation", params={
            "latitude": ",".join(f"{v:.2f}" for v in la[i:i+100]), "longitude": ",".join(f"{v:.2f}" for v in lo[i:i+100])}, timeout=60)
        r.raise_for_status(); el += r.json()["elevation"]; time.sleep(1)
    np.savez_compressed(ef, elev=np.array(el, float).reshape(LA.shape), lat=lat, lon=lon)
    print("elevation ok")

for name, u in {"oni.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
                "pdo.txt": "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"}.items():
    if not (OUT / name).exists():
        r = requests.get(u, timeout=60); r.raise_for_status(); (OUT / name).write_bytes(r.content); print(name, "ok")
