"""Download global ComCat M>=4.7, 1973-2012, yearly CSV chunks. Usage: python3 -I download.py OUTDIR"""
import sys, os, time, requests
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
for y in range(1973, 2013):
    for half in (0, 1):
        fn = os.path.join(out, f"comcat_{y}_{half}.csv")
        if os.path.exists(fn) and os.path.getsize(fn) > 1000:
            continue
        st = f"{y}-01-01" if half == 0 else f"{y}-07-01"
        en = f"{y}-07-01" if half == 0 else f"{y+1}-01-01"
        params = dict(format="csv", starttime=st, endtime=en, minmagnitude=4.7, eventtype="earthquake",
                      orderby="time-asc", limit=20000)
        for attempt in range(5):
            try:
                r = requests.get(URL, params=params, timeout=120)
                r.raise_for_status()
                txt = r.text
                if not txt.startswith("time,"):
                    raise ValueError("bad header")
                with open(fn + ".tmp", "w") as f:
                    f.write(txt)
                os.replace(fn + ".tmp", fn)
                print(y, half, txt.count("\n") - 1, flush=True)
                break
            except Exception as e:
                print("retry", y, half, e, flush=True); time.sleep(5 * (attempt + 1))
        time.sleep(0.5)
