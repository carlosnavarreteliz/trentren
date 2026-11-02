"""Download MUSTANG daily PDF-mode noise time series (noise-mode-timeseries) for Chilean broadband stations.

Usage: /opt/miniconda3/bin/python3 -I download.py <out_dir> <channels_region.txt>
Caches one raw text file per target in out_dir/raw/. Re-running skips cached targets.
"""
import sys, time, re
from pathlib import Path
import requests

out = Path(sys.argv[1]); (out / "raw").mkdir(parents=True, exist_ok=True)
chan_file = Path(sys.argv[2])

URL = "https://service.iris.edu/mustang/noise-mode-timeseries/1/query"

# candidate stations: every BHZ station in the region listing from networks that MUSTANG covers here
stas = set()
for line in chan_file.read_text().splitlines()[1:]:
    f = line.split("|")
    if len(f) < 17 or f[3] != "BHZ":
        continue
    if f[0] in ("IU", "G", "GT", "C", "C1"):
        stas.add((f[0], f[1], f[2] if f[2] else "--"))
# a few explicit extra location codes MUSTANG uses
for t in [("IU", "LVC", "00"), ("IU", "LVC", "10"), ("IU", "LCO", "--"), ("IU", "LCO", "00"), ("IU", "LCO", "10"),
          ("G", "PEL", "--"), ("G", "PEL", "00"), ("G", "COYC", "00"), ("G", "COYC", "10"),
          ("GT", "PLCA", "--"), ("GT", "PLCA", "00")]:
    stas.add(t)

print(len(stas), "targets")
s = requests.Session()
for net, sta, loc in sorted(stas):
    tgt = f"{net}.{sta}.{loc}.BHZ.M"
    fn = out / "raw" / f"{tgt}.txt"
    if fn.exists():
        continue
    params = dict(target=tgt, starttime="1995-01-01", endtime="2026-10-18", format="text", **{"text.style": "table"})
    for attempt in range(3):
        try:
            r = s.get(URL, params=params, timeout=300)
            break
        except Exception as e:  # noqa
            print("retry", tgt, e); time.sleep(5)
    else:
        continue
    if r.status_code == 200 and r.text.startswith("target:"):
        fn.write_text(r.text)
        nd = len(re.findall(r"^\d{4}-\d\d-\d\d,", r.text, flags=re.M))
        print(tgt, "days", nd)
    else:
        fn.write_text("")  # mark as no data
        print(tgt, "no data", r.status_code)
    time.sleep(0.5)
