"""Percepción: sismos (CSN vía api.xor.cl + USGS ComCat) y noticias (Google Noticias RSS)."""
import html
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from .config import UA


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_csn():
    """Últimos sismos reportados por el Centro Sismológico Nacional (espejo api.xor.cl)."""
    data = json.loads(_get("https://api.xor.cl/sismo/recent"))
    out = []
    for e in data.get("events", []):
        try:
            t = datetime.strptime(e["utc_date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            out.append({
                "id": "csn-" + str(e["id"]), "fuente": "CSN", "t": t.isoformat(),
                "lat": float(e["latitude"]), "lon": float(e["longitude"]), "prof": float(e["depth"]),
                "mag": float(e["magnitude"]["value"]), "tipo_mag": e["magnitude"].get("measure_unit", ""),
                "lugar": e.get("geo_reference", ""), "url": e.get("url", ""),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _usgs_geojson(box, days, min_mag):
    a, b, c, d = box
    end = datetime.now(timezone.utc)
    q = {"format": "geojson", "starttime": (end - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S"),
         "minlatitude": a, "maxlatitude": b, "minlongitude": c, "maxlongitude": d,
         "minmagnitude": min_mag, "orderby": "time"}
    return json.loads(_get("https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(q)))


def fetch_usgs(boxes, days=7, min_mag=2.5):
    """Sismos recientes USGS (para mostrar), en una o más cajas lat/lon."""
    out = []
    for box in boxes:
        for f in _usgs_geojson(box, days, min_mag).get("features", []):
            p, c = f["properties"], f["geometry"]["coordinates"]
            if p.get("mag") is None or p.get("type", "earthquake") != "earthquake":
                continue
            out.append({
                "id": "us-" + f["id"], "fuente": "USGS",
                "t": datetime.fromtimestamp(p["time"] / 1000, tz=timezone.utc).isoformat(),
                "lat": float(c[1]), "lon": float(c[0]), "prof": float(c[2] if c[2] is not None else 30.0),
                "mag": float(p["mag"]), "tipo_mag": p.get("magType", ""), "lugar": p.get("place", ""),
                "url": p.get("url", ""), "tsunami": int(p.get("tsunami") or 0), "sentido": p.get("felt"),
            })
    return out


def fetch_usgs_cat(boxes, days=30, min_mag=4.0):
    """Catálogo vivo M>=4 en formato del modelo (time naive UTC, latitude, longitude, depth, mag, magType)."""
    import pandas as pd
    rows = [{"time": datetime.fromisoformat(e["t"]).replace(tzinfo=None), "latitude": e["lat"], "longitude": e["lon"],
             "depth": e["prof"], "mag": e["mag"], "magType": e["tipo_mag"] or ""} for e in fetch_usgs(boxes, days, min_mag)]
    return pd.DataFrame(rows, columns=["time", "latitude", "longitude", "depth", "mag", "magType"])


def _strip(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def search_news(query, days=2, limit=12, raw=False, loc=("es-419", "CL", "CL:es-419")):
    """Titulares de Google Noticias (es-419, Chile). raw=True: la consulta trae sus propios
    operadores de fecha (after:/before:) para revivir la prensa de un sismo histórico."""
    q = query if raw else f"{query} when:{max(1, int(days))}d"
    url = ("https://news.google.com/rss/search?" +
           urllib.parse.urlencode({"q": q, "hl": loc[0], "gl": loc[1], "ceid": loc[2]}))
    root = ET.fromstring(_get(url))
    items = []
    for it in root.iter("item"):
        src = it.find("source")
        items.append({
            "titulo": _strip(it.findtext("title")),
            "medio": src.text if src is not None else "",
            "fecha": it.findtext("pubDate", ""),
            "url": it.findtext("link", ""),
        })
        if len(items) >= limit:
            break
    return items
