"""Download continuous BH? (20 Hz) miniSEED for IPOC (CX) stations from GEOFON FDSN dataselect.

Usage: /opt/miniconda3/bin/python3 -I download.py <out_dir> [max_gb]
One file per station-day: <out_dir>/<window>/<STA>/<YYYY-MM-DD>.mseed . Re-running skips cached days.
Stops when cumulative bytes on disk exceed max_gb (default 6.0).
Day order is interleaved across windows (newest-to-oldest relative to window end) so that, if the cap
is hit, every window loses the same (earliest) days.
"""
import sys, time, datetime as dt
from pathlib import Path
import requests

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
max_gb = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
URL = "https://geofon.gfz.de/fdsnws/dataselect/1/query"
STATIONS = ["PSGCX", "PB11", "PATCX"]
# windows: (name, first day, last day inclusive). Last day is truncated at 23:46 UTC in processing.
WINDOWS = [("T2014", dt.date(2013, 12, 22), dt.date(2014, 4, 1)),
           ("C2013", dt.date(2012, 12, 22), dt.date(2013, 4, 1)),
           ("C2017", dt.date(2016, 12, 22), dt.date(2017, 4, 1))]
NDAYS = 101


def total_bytes():
    return sum(f.stat().st_size for f in out.rglob("*.mseed"))


s = requests.Session()
tot = total_bytes()
for k in range(NDAYS):  # k days before window end
    for wname, d0, d1 in WINDOWS:
        day = d1 - dt.timedelta(days=k)
        for sta in STATIONS:
            fn = out / wname / sta / f"{day.isoformat()}.mseed"
            if fn.exists():
                continue
            fn.parent.mkdir(parents=True, exist_ok=True)
            params = dict(net="CX", sta=sta, cha="BH?", start=f"{day.isoformat()}T00:00:00",
                          end=f"{(day + dt.timedelta(days=1)).isoformat()}T00:00:00")
            for attempt in range(4):
                try:
                    r = s.get(URL, params=params, timeout=300)
                    if r.status_code in (200, 204):
                        break
                except Exception as e:  # noqa
                    print("retry", sta, day, e, flush=True)
                time.sleep(5 * (attempt + 1))
            else:
                print("FAILED", wname, sta, day, flush=True)
                continue
            data = r.content if r.status_code == 200 else b""
            fn.write_bytes(data)
            tot += len(data)
            print(wname, sta, day, len(data), f"tot={tot/1e9:.2f}GB", flush=True)
            if tot > max_gb * 1e9:
                print("CAP REACHED", flush=True); sys.exit(0)
print("DONE", tot / 1e9, flush=True)
