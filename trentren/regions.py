"""Países que vigila Trentren: Chile, Japón y Estados Unidos.

Cada país tiene su catálogo USGS (cajas geográficas), sus zonas, sus ciudades, sus
autoridades y su propio modelo de escalamiento entrenado con sus sismos pasados
(mismo protocolo que Chile: train <=2020, Platt cross-fitted 2021-23, holdout 2024+).

  python3 -m trentren.regions --fetch japon     # descarga catálogo 1990-2026
  python3 -m trentren.regions --train japon     # entrena modelo v4 del país
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import ROOT, UA

sys.path.insert(0, str(ROOT))
import forecast_short as F  # noqa: E402

RAW = ROOT / "data" / "raw"
RES = ROOT / "results"

PAISES = {
    "chile": {
        "nombre": {"es": "Chile", "en": "Chile", "ja": "チリ"},
        "cajas": [(-56.0, -17.0, -78.0, -66.0)],
        "vista": (-80.0, -60.5, -56.4, -16.6),
        "catalogo": RAW / "chile_usgs_1900_2026.csv",
        "modelo": RES / "short_term_model_v4.json",
        "tz": "America/Santiago", "idioma": "es", "noticias": ("es-419", "CL", "CL:es-419"),
        "consulta": "sismo Chile",
        "autoridades": ["SENAPRED", "SHOA", "Centro Sismológico Nacional"],
        "links": [("SENAPRED", "https://senapred.cl"), ("SHOA", "https://www.shoa.cl"), ("CSN", "https://www.sismologia.cl")],
        "tsunami": "Si estás en la costa y un sismo te impide mantenerte en pie, evacúa a la zona segura sin esperar la alerta.",
        "zonas": [(n, a, b, -78.0, -66.0) for n, a, b in F.ZONES],
    },
    "japon": {
        "nombre": {"es": "Japón", "en": "Japan", "ja": "日本"},
        "cajas": [(24.0, 46.0, 122.0, 150.0)],
        "vista": (122.0, 150.0, 23.5, 46.5),
        "catalogo": RAW / "japon_usgs_1990_2026.csv",
        "modelo": RES / "short_term_model_v4_japon.json",
        "tz": "Asia/Tokyo", "idioma": "ja", "noticias": ("ja", "JP", "JP:ja"),
        "consulta": "地震",
        "autoridades": ["気象庁 (JMA)", "内閣府防災", "自治体"],
        "links": [("気象庁 JMA", "https://www.jma.go.jp"), ("内閣府防災", "https://www.bousai.go.jp")],
        "tsunami": "強い揺れや長い揺れを感じたら、津波警報を待たずに直ちに高台へ避難してください。",
        "zonas": [("Hokkaido", 41.3, 46.0, 128.5, 150.0), ("Tohoku", 37.0, 41.3, 138.5, 150.0),
                  ("Chubu-Hokuriku", 35.6, 41.3, 128.5, 138.5), ("Kanto-Izu", 24.0, 37.0, 138.5, 150.0),
                  ("Nankai", 24.0, 35.6, 132.0, 138.5), ("Kyushu", 30.5, 35.6, 122.0, 132.0),
                  ("Ryukyu", 24.0, 30.5, 122.0, 132.0)],
    },
    "eeuu": {
        "nombre": {"es": "Estados Unidos", "en": "United States", "ja": "アメリカ合衆国"},
        "cajas": [(31.0, 50.0, -128.0, -112.0), (50.0, 72.0, -180.0, -129.0), (18.0, 23.0, -161.0, -154.0)],
        "vista": (-128.5, -111.5, 31.0, 50.0),
        "catalogo": RAW / "eeuu_usgs_1990_2026.csv",
        "modelo": RES / "short_term_model_v4_eeuu.json",
        "tz": "America/Los_Angeles", "idioma": "en", "noticias": ("en-US", "US", "US:en"),
        "consulta": "earthquake",
        "autoridades": ["USGS", "NOAA Tsunami Warning Centers", "FEMA", "ShakeAlert"],
        "links": [("USGS", "https://earthquake.usgs.gov"), ("NOAA Tsunami", "https://www.tsunami.gov"), ("FEMA", "https://www.ready.gov/earthquakes")],
        "tsunami": "If you are on the coast and feel strong or long shaking, move to high ground immediately; don't wait for an official warning.",
        "zonas": [("Pacific Northwest", 42.0, 50.0, -128.0, -116.5), ("Northern California", 37.5, 42.0, -128.0, -118.0),
                  ("Southern California", 31.0, 37.5, -122.0, -114.0), ("Basin and Range", 31.0, 50.0, -118.0, -112.0),
                  ("Aleutians", 50.0, 56.0, -180.0, -163.0), ("Alaska", 50.0, 72.0, -180.0, -129.0),
                  ("Hawaii", 18.0, 23.0, -161.0, -154.0)],
    },
}


def zona(pais, lat, lon):
    for n, a, b, c, d in PAISES[pais]["zonas"]:
        if a <= lat < b and c <= lon < d:
            return n
    return "Otra"


def en_pais(pais, lat, lon):
    return any(a <= lat <= b and c <= lon <= d for a, b, c, d in PAISES[pais]["cajas"])


# ---------------------------------------------------------------- catálogos
def _usgs(params):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(params)
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120) as r:
                return r.read().decode()
        except Exception as e:
            print("  reintento", k, e, flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError("USGS no responde")


def fetch(pais, y0=1990, y1=None):
    """Catálogo USGS M>=4 por año (CSV, mismas columnas que el de Chile) + M>=7 desde 1900."""
    y1 = y1 or datetime.now(timezone.utc).year
    P = PAISES[pais]
    parts = []
    for a, b, c, d in P["cajas"]:
        for y in range(y0, y1 + 1):
            for half in ((f"{y}-01-01", f"{y}-07-01"), (f"{y}-07-01", f"{y + 1}-01-01")):
                txt = _usgs({"format": "csv", "starttime": half[0], "endtime": half[1], "minmagnitude": 4.0,
                             "minlatitude": a, "maxlatitude": b, "minlongitude": c, "maxlongitude": d, "orderby": "time-asc"})
                lines = txt.strip().splitlines()
                if len(lines) > 1:
                    parts.append(pd.read_csv(pd.io.common.StringIO(txt)))
            print(f"  {pais} caja {a},{c} año {y}: ok", flush=True)
        txt = _usgs({"format": "csv", "starttime": "1900-01-01", "endtime": f"{y0}-01-01", "minmagnitude": 7.0,
                     "minlatitude": a, "maxlatitude": b, "minlongitude": c, "maxlongitude": d})
        if len(txt.strip().splitlines()) > 1:
            parts.append(pd.read_csv(pd.io.common.StringIO(txt)))
    df = pd.concat(parts, ignore_index=True).drop_duplicates("id").sort_values("time")
    df.to_csv(P["catalogo"], index=False)
    print(f"{pais}: {len(df)} eventos -> {P['catalogo']}", flush=True)


def load(pais):
    P = PAISES[pais]
    if pais == "chile":
        return F.load_local()
    d = pd.read_csv(P["catalogo"], usecols=["time", "latitude", "longitude", "depth", "mag", "magType", "type"])
    d = d[d["type"] == "earthquake"]
    d["time"] = pd.to_datetime(d["time"], format="ISO8601", utc=True).dt.tz_convert(None)
    d = d.dropna(subset=["mag"]).sort_values("time").reset_index(drop=True)
    d["depth"] = d["depth"].fillna(30.0)
    d["mag_raw"] = d["mag"]
    d["mag"] = [F.to_mw(m, t) for m, t in zip(d.mag, d.magType.fillna(""))]
    d["tmin"] = (d.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
    return d


# ---------------------------------------------------------------- entrenamiento
def train(pais):
    """Mismo protocolo que forecast_short.train_v4, sobre el catálogo del país."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler
    P = PAISES[pais]
    d = load(pais)
    print(f"{pais}: {len(d)} eventos; construyendo triggers...", flush=True)
    df = F.build_dataset(d)
    tr = df[df.time < "2021-01-01"].reset_index(drop=True)
    ca = df[(df.time >= "2021-01-01") & (df.time < "2024-01-01")].reset_index(drop=True)
    ho = df[df.time >= "2024-01-01"].reset_index(drop=True)
    cols = F.COLS_V2
    sc = StandardScaler().fit(tr[cols].to_numpy())
    Xtr, Xca, Xho = (sc.transform(s[cols].to_numpy()) for s in (tr, ca, ho))
    model = {"pais": pais, "cols": cols, "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(), "horizons": {},
             "trained_at": datetime.now(timezone.utc).isoformat(), "n_train": len(tr), "n_calib": len(ca), "n_holdout": len(ho),
             "train_period": f"{df.time.min().year}-2020", "calibration": "Platt cross-fitted 5 bloques 2021-2023; umbral q95 cross-fitted"}
    for h in F.HORIZONS_MIN:
        yc, yh = ca[f"y_{h}"].to_numpy(), ho[f"y_{h}"].to_numpy()
        clf = LogisticRegression(C=1.0, max_iter=500).fit(Xtr, tr[f"y_{h}"].to_numpy())
        s_ca, s_ho = clf.decision_function(Xca), clf.decision_function(Xho)
        p_cv = np.zeros(len(ca))
        for f in np.array_split(np.arange(len(ca)), 5):
            rest = np.setdiff1d(np.arange(len(ca)), f)
            p_cv[f] = LogisticRegression(C=1e6).fit(s_ca[rest].reshape(-1, 1), yc[rest]).predict_proba(s_ca[f].reshape(-1, 1))[:, 1]
        platt = LogisticRegression(C=1e6).fit(s_ca.reshape(-1, 1), yc)
        p_ho = platt.predict_proba(s_ho.reshape(-1, 1))[:, 1]
        thr = float(np.quantile(p_cv, 0.95))
        pred = p_ho >= thr
        tp = int((pred & (yh == 1)).sum())
        model["horizons"][str(h)] = {
            "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
            "platt_coef": float(platt.coef_[0][0]), "platt_intercept": float(platt.intercept_[0]),
            "threshold": thr, "base_rate_calib": float(yc.mean()),
            "holdout": {"n": len(ho), "pos": int(yh.sum()),
                        "auc": float(roc_auc_score(yh, p_ho)) if 0 < yh.sum() < len(yh) else None,
                        "pr_auc": float(average_precision_score(yh, p_ho)) if yh.sum() else None,
                        "base_rate": float(yh.mean()), "alerts": int(pred.sum()), "hits": tp,
                        "precision": float(tp / max(pred.sum(), 1)), "recall": float(tp / max(yh.sum(), 1))}}
        hh = model["horizons"][str(h)]["holdout"]
        print(f"[{pais} {F.H_LABEL[h]}] n_train={len(tr)} AUC={hh['auc']} P={hh['precision']:.2f} R={hh['recall']:.2f} "
              f"base={hh['base_rate']:.3f} alertas={hh['alerts']}", flush=True)
    P["modelo"].write_text(json.dumps(model, indent=1))
    print(f"modelo -> {P['modelo']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch")
    ap.add_argument("--train")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.fetch)
    if a.train:
        train(a.train)
