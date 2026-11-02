"""Fetch ERA5 daily mean surface pressure (Open-Meteo archive) at one offshore forearc point per 1-deg bin.
Usage: python -I fetch_era5.py <out_dir>
"""
import sys, os, time, json
import numpy as np
import requests

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

# approximate Chilean coastline longitude vs latitude (piecewise linear), offshore point = coast - 0.75 deg
COAST = [(-17.0, -70.6), (-18.5, -70.35), (-21.0, -70.15), (-23.5, -70.45), (-26.0, -70.7), (-28.0, -71.2),
         (-30.0, -71.4), (-32.0, -71.5), (-33.5, -71.7), (-35.0, -72.2), (-36.5, -72.9), (-37.5, -73.6),
         (-39.0, -73.3), (-41.0, -73.9), (-43.0, -74.2), (-46.0, -75.5)]
lat_c = np.arange(-45.5, -17.0, 1.0)  # bin centres, bin 0 = -45.5
cl = np.array(COAST)
coast_lon = np.interp(lat_c, cl[::-1, 0], cl[::-1, 1])
pts = [(float(la), float(round(lo - 0.75, 2))) for la, lo in zip(lat_c, coast_lon)]
json.dump(pts, open(os.path.join(out_dir, "era5_points.json"), "w"))

for b, (la, lo) in enumerate(pts):
    fn = os.path.join(out_dir, f"era5_sp_bin{b:02d}.json")
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        continue
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={la}&longitude={lo}&start_date=1974-01-01&end_date=2026-10-15"
           "&daily=surface_pressure_mean&timezone=GMT&models=era5")
    for attempt in range(5):
        r = requests.get(url, timeout=180)
        if r.status_code == 200:
            open(fn, "w").write(r.text)
            print("ok", b, la, lo, flush=True)
            break
        print("retry", b, r.status_code, r.text[:200], flush=True)
        time.sleep(30 * (attempt + 1))
    time.sleep(3)
