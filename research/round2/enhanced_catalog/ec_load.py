"""Loaders for the high-resolution catalogs (untrusted input: parsed as plain numbers only)."""
import numpy as np, pandas as pd

def load_ipoc(path):
    cols = ["yr","mo","dy","hh","mi","ss","lat","lon","depth","ml","ml_sd","ma","ma_sd","cls"]
    d = pd.read_csv(path, comment="#", sep=r"\s+", header=None, names=cols)
    sec = d.ss.astype(float)
    t = pd.to_datetime(dict(year=d.yr, month=d.mo, day=d.dy, hour=d.hh, minute=d.mi)) + pd.to_timedelta(sec, unit="s")
    out = pd.DataFrame({"time": t, "latitude": d.lat.astype(float), "longitude": d.lon.astype(float),
                        "depth": d.depth.astype(float), "mag": d.ml.astype(float), "cls": d.cls.astype(str)})
    return out.sort_values("time").reset_index(drop=True)

def load_riedel(path):
    d = pd.read_csv(path, usecols=["Date","Time","Lat","Lon","Dep","Ml","Type"])
    t = pd.to_datetime(d.Date + " " + d.Time, errors="coerce")
    out = pd.DataFrame({"time": t, "latitude": d.Lat.astype(float), "longitude": d.Lon.astype(float),
                        "depth": -d.Dep.astype(float), "mag": pd.to_numeric(d.Ml, errors="coerce"), "cls": d.Type.astype(str)})
    return out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
