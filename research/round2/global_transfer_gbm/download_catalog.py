"""Download the USGS ComCat global catalog, M>=4.7, eventtype=earthquake, 1973-01-01 .. today, yearly chunks.
Usage: python3 -I download_catalog.py <out_dir>
Writes <out_dir>/global_m47_<year>.csv (raw FDSN csv) and <out_dir>/download_log.json.
"""
import sys, os, time, json
import requests

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
log = {}
for year in range(1973, 2027):
    fn = os.path.join(out, f"global_m47_{year}.csv")
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        continue
    params = dict(format="csv", starttime=f"{year}-01-01", endtime=f"{year + 1}-01-01", minmagnitude=4.7,
                  eventtype="earthquake", orderby="time-asc", limit=20000)
    for attempt in range(5):
        try:
            r = requests.get(URL, params=params, timeout=300)
            if r.status_code == 200:
                with open(fn + ".tmp", "w") as f:
                    f.write(r.text)
                os.replace(fn + ".tmp", fn)
                n = r.text.count("\n") - 1
                log[year] = n
                print(year, n, flush=True)
                if n >= 19999:
                    print("WARNING: hit limit", year, flush=True)
                break
            print(year, "status", r.status_code, flush=True)
        except Exception as e:  # noqa
            print(year, "err", e, flush=True)
        time.sleep(5 * (attempt + 1))
    time.sleep(1)
json.dump({"downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "counts": log},
          open(os.path.join(out, "download_log.json"), "w"), indent=1)
