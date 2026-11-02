#!/usr/bin/env python3
"""forecast_short.py — Pronóstico operativo de escalamiento sísmico a minutos-horas.

Ante cada sismo M>=4.7, estima P(otro M>=5.0 dentro de H y 150 km) para
H = 60 min, 3 h y 24 h. Modelos logisticos calibrados (Platt) sobre features
causales de clustering. Entrena con el catalogo local USGS.

Uso:
  python3 forecast_short.py --train              # entrena, calibra y backtestea (guarda JSON)
  python3 forecast_short.py                      # pronostico con catalogo local (ultimos triggers)
  python3 forecast_short.py --fetch-live         # actualiza con USGS en vivo y pronostica
  python3 forecast_short.py --fetch-live --n 5   # muestra ultimos 5 triggers

El JSON del modelo queda en results/short_term_model.json.
"""
import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
CAT = ROOT / "data" / "raw" / "chile_usgs_1900_2026.csv"
MODEL_JSON = ROOT / "results" / "short_term_model.json"
MODEL_JSON_V2 = ROOT / "results" / "short_term_model_v2.json"
MODEL_JSON_V3 = ROOT / "results" / "short_term_model_v3.json"
MODEL_JSON_V4 = ROOT / "results" / "short_term_model_v4.json"
# v2 (E3): features con split interfaz/intraslab + memoria 3h. v1 = 4 cols legacy.
COLS_V2 = ["x1i", "x1s", "x2", "x3", "x4i", "x4s", "x5"]
COLS_V1 = ["x1", "x2", "x3", "x4"]

HORIZONS_MIN = [60, 180, 1440]
H_LABEL = {60: "60min", 180: "3h", 1440: "24h"}
R_KM = 6371.0

ZONES = [  # (nombre, lat_min, lat_max) en grados (negativos)
    ("Arica-Parinacota", -19.5, -17.0),
    ("Iquique-Loa", -21.5, -19.5),
    ("Antofagasta-Calama", -25.0, -21.5),
    ("Atacama-Copiapo", -29.0, -25.0),
    ("Coquimbo-Illapel", -32.2, -29.0),
    ("Valparaiso-RM", -34.5, -32.2),
    ("OHiggins-Maule-Biobio", -38.5, -34.5),
    ("Araucania-Valdivia-Chiloe", -44.0, -38.5),
    ("Austral", -56.0, -44.0),
]


def zone_of(lat):
    for name, a, b in ZONES:
        if a <= lat < b:
            return name
    return "Fuera"


def hav_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def to_mw(mag, mag_type):
    """Homogeneiza a Mw-equivalente (Scordilis 2006). Mw-familia/ml/md: sin cambio.
    v3: el catalogo mixto USGS (78% mb en 2024+) etiqueta mal la poblacion M5:
    mb 4.7 ~ Mw 5.0. Con etiquetas homogeneas el mismo modelo rinde AUC 0.87."""
    t = str(mag_type).lower()
    if t == "mb":
        return 0.85 * mag + 1.03
    if t == "ms":
        return 0.67 * mag + 2.07 if mag <= 6.1 else 0.99 * mag + 0.08
    return mag


def load_local():
    d = pd.read_csv(CAT, usecols=["time", "latitude", "longitude", "depth", "mag", "magType", "type"])
    d = d[d["type"] == "earthquake"]
    d["time"] = pd.to_datetime(d["time"], format="ISO8601", utc=True).dt.tz_convert(None)
    d = d[d.latitude.between(-56, -17) & d.longitude.between(-78, -66)]
    d = d.dropna(subset=["mag"]).sort_values("time").reset_index(drop=True)
    d["mag_raw"] = d["mag"]
    d["mag"] = [to_mw(m, t) for m, t in zip(d.mag, d.magType.fillna(""))]
    d["tmin"] = (d.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
    return d


def fetch_live_usgs(days=90, min_mag=4.0):
    end_dt = datetime.now(timezone.utc)
    start_dt = pd.Timestamp(end_dt) - pd.Timedelta(days=days)
    params = {"format": "geojson", "starttime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
              "endtime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"), "minlatitude": -56.0,
              "maxlatitude": -17.0, "minlongitude": -78.0, "maxlongitude": -66.0,
              "minmagnitude": min_mag, "orderby": "time-asc"}
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Chile-ShortTerm-Forecast/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rows = []
    for f in data.get("features", []):
        p, g = f.get("properties", {}), f.get("geometry", {})
        c = g.get("coordinates", [None, None, None])
        if p.get("mag") is None or c[0] is None:
            continue
        dt = datetime.fromtimestamp(p["time"] / 1000.0, tz=timezone.utc).replace(tzinfo=None)
        rows.append({"time": dt, "latitude": float(c[1]), "longitude": float(c[0]),
                     "depth": float(c[2]) if c[2] is not None else 30.0,
                     "mag": float(to_mw(float(p["mag"]), p.get("magType", "")))})
    d = pd.DataFrame(rows)
    if len(d):
        d["tmin"] = (d.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
        d = d.sort_values("time").reset_index(drop=True)
    return d


def trigger_features(r, i0, tm, la, lo, mg, dp):
    """Features causales v2 (E3): split interfaz/intraslab + memoria 3h."""
    lo24 = np.searchsorted(tm, r.tmin - 1440, side="left")
    past = np.arange(lo24, i0)
    dd = hav_km(r.latitude, r.longitude, la[past], lo[past]) if len(past) else np.array([])
    near = past[dd <= 150]
    dt = r.tmin - tm[near]
    m_near = mg[near]
    z_near = dp[near]
    m47 = m_near >= 4.7
    iface = z_near <= 60
    return {
        "x1i": float(np.log1p(np.sum((dt <= 60) & m47 & iface))),
        "x1s": float(np.log1p(np.sum((dt <= 60) & m47 & ~iface))),
        "x2": float(r.mag),
        "x3": float(np.log10(1 + dt[m47].min())) if np.any(m47) else float(np.log10(1 + 1440)),
        "x4i": float(m_near[m47 & iface].max()) if np.any(m47 & iface) else 4.2,
        "x4s": float(m_near[m47 & ~iface].max()) if np.any(m47 & ~iface) else 4.2,
        "x5": float(np.log1p(np.sum((dt <= 180) & m47))),
    }


def build_dataset(d):
    """Triggers M>=4.7 con features causales y targets por horizonte."""
    tm = d.tmin.to_numpy()
    la, lo, mg, dp = (d[c].to_numpy() for c in ["latitude", "longitude", "mag", "depth"])
    trig = d[(d.mag >= 4.7) & (d.time >= "1990-01-01")].reset_index()
    rows = []
    for _, r in trig.iterrows():
        i0 = int(r["index"])
        f = trigger_features(r, i0, tm, la, lo, mg, dp)
        row = {"time": r.time, "lat": r.latitude, "lon": r.longitude, **f}
        for h in HORIZONS_MIN:
            hi = np.searchsorted(tm, r.tmin + h, side="right")
            fut = np.arange(i0 + 1, hi)
            y = 0
            if len(fut):
                cand = fut[mg[fut] >= 5.0]
                if len(cand):
                    y = int((hav_km(r.latitude, r.longitude, la[cand], lo[cand]) <= 150).any())
            row[f"y_{h}"] = y
        rows.append(row)
    return pd.DataFrame(rows)


def ll_bern(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log1p(-p)


def train():
    print("cargando catalogo y construyendo dataset (1-3 min)...", flush=True)
    d = load_local()
    df = build_dataset(d)
    tr = df[df.time < "2021-01-01"].reset_index(drop=True)
    ca = df[(df.time >= "2021-01-01") & (df.time < "2024-01-01")].reset_index(drop=True)
    ho = df[df.time >= "2024-01-01"].reset_index(drop=True)
    cols = COLS_V2
    sc = StandardScaler().fit(tr[cols].to_numpy())
    Xtr, Xca, Xho = (sc.transform(s[cols].to_numpy()) for s in (tr, ca, ho))
    model = {"cols": cols, "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
             "horizons": {}, "trained_at": datetime.now(timezone.utc).isoformat(),
             "n_train": len(tr), "n_calib": len(ca), "n_holdout": len(ho)}
    print(f"train={len(tr)} calib={len(ca)} holdout={len(ho)}")
    for h in HORIZONS_MIN:
        yc = ca[f"y_{h}"].to_numpy()
        yh = ho[f"y_{h}"].to_numpy()
        clf = LogisticRegression(C=1.0).fit(Xtr, tr[f"y_{h}"].to_numpy())
        s_ca = clf.decision_function(Xca).reshape(-1, 1)
        s_ho = clf.decision_function(Xho).reshape(-1, 1)
        platt = LogisticRegression(C=1e6).fit(s_ca, yc)
        p_ca = platt.predict_proba(s_ca)[:, 1]
        p_ho = platt.predict_proba(s_ho)[:, 1]
        # umbral backtesteado en calibracion: percentil 95 de las probabilidades
        # de calibracion (presupuesto de alerta 5%; interpretable para politica).
        # v3: F0.5 daba umbrales tan altos que 3h/24h nunca alertaban.
        thr = float(np.quantile(p_ca, 0.95))
        pred_ho = p_ho >= thr
        tp = int((pred_ho & (yh == 1)).sum())
        base = yh.mean()
        mago = LogisticRegression(C=1.0).fit(Xtr[:, [cols.index("x2")]], tr[f"y_{h}"].to_numpy())
        p_mago = mago.predict_proba(Xho[:, [cols.index("x2")]])[:, 1]
        ig = (ll_bern(yh, p_ho) - ll_bern(yh, p_mago)).sum() / max(yh.sum(), 1)
        model["horizons"][str(h)] = {
            "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
            "platt_coef": float(platt.coef_[0][0]), "platt_intercept": float(platt.intercept_[0]),
            "threshold": thr, "base_rate_calib": float(yc.mean()),
            "holdout": {"n": len(ho), "pos": int(yh.sum()), "ig_vs_magonly": float(ig),
                        "auc": float(roc_auc_score(yh, p_ho)) if yh.sum() > 0 else None,
                        "pr_auc": float(average_precision_score(yh, p_ho)) if yh.sum() > 0 else None,
                        "base_rate": float(base),
                        "alerts": int(pred_ho.sum()), "hits": tp,
                        "precision": float(tp / max(pred_ho.sum(), 1)),
                        "recall": float(tp / max(yh.sum(), 1))},
        }
        hh = model["horizons"][str(h)]["holdout"]
        print(f"[{H_LABEL[h]}] holdout n={hh['n']} pos={hh['pos']} IG-vs-mag={hh['ig_vs_magonly']:+.3f} "
              f"AUC={hh['auc']} PR-AUC={hh['pr_auc']} base={hh['base_rate']:.3f} "
              f"alertas={hh['alerts']} aciertos={hh['hits']} P={hh['precision']:.2f} R={hh['recall']:.2f} thr={thr:.2f}")
    model["mag_scale"] = "Mw-equiv Scordilis (v3)"
    MODEL_JSON_V3.parent.mkdir(parents=True, exist_ok=True)
    MODEL_JSON_V3.write_text(json.dumps(model, indent=1))
    print(f"modelo guardado en {MODEL_JSON_V3}")


def calibration_stats(y, p):
    """Pendiente/intercepto de calibración (y ~ logit(p)), ECE y pred-vs-obs."""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    try:
        lr = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        slope, inter = float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        slope, inter = None, None
    bins = np.quantile(p, np.linspace(0, 1, 11)); bins[0], bins[-1] = -1e-9, 1 + 1e-9
    idx = np.digitize(p, bins[1:-1]); ece = 0.0
    for b in range(10):
        m = idx == b
        if m.sum():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"cal_slope": slope, "cal_intercept": inter, "ece": float(ece),
            "pred_mean": float(p.mean()), "obs_rate": float(y.mean()),
            "brier": float(np.mean((p - y) ** 2))}


def train_v4():
    """v4 = mismo score que v3 (logit <=2020) pero calibración honesta:
    Platt cross-fitted en calib 2021-23 (5 bloques temporales), umbral desde
    probs cross-fitted, y diagnóstico de calibración en holdout 2024+."""
    from sklearn.isotonic import IsotonicRegression
    print("v4: cargando y construyendo dataset...", flush=True)
    d = load_local(); df = build_dataset(d)
    tr = df[df.time < "2021-01-01"].reset_index(drop=True)
    ca = df[(df.time >= "2021-01-01") & (df.time < "2024-01-01")].reset_index(drop=True)
    ho = df[df.time >= "2024-01-01"].reset_index(drop=True)
    cols = COLS_V2
    sc = StandardScaler().fit(tr[cols].to_numpy())
    Xtr, Xca, Xho = (sc.transform(s[cols].to_numpy()) for s in (tr, ca, ho))
    model = {"cols": cols, "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
             "horizons": {}, "trained_at": datetime.now(timezone.utc).isoformat(),
             "n_train": len(tr), "n_calib": len(ca), "n_holdout": len(ho),
             "calibration": "Platt cross-fitted 5 bloques temporales en 2021-2023; umbral q95 cross-fitted"}
    for h in HORIZONS_MIN:
        yc, yh = ca[f"y_{h}"].to_numpy(), ho[f"y_{h}"].to_numpy()
        clf = LogisticRegression(C=1.0).fit(Xtr, tr[f"y_{h}"].to_numpy())
        s_ca = clf.decision_function(Xca); s_ho = clf.decision_function(Xho)
        # cross-fitting temporal: 5 bloques, Platt entrenado fuera del bloque
        p_cv = np.zeros(len(ca)); folds = np.array_split(np.arange(len(ca)), 5)
        for f in folds:
            rest = np.setdiff1d(np.arange(len(ca)), f)
            pl = LogisticRegression(C=1e6).fit(s_ca[rest].reshape(-1, 1), yc[rest])
            p_cv[f] = pl.predict_proba(s_ca[f].reshape(-1, 1))[:, 1]
        platt = LogisticRegression(C=1e6).fit(s_ca.reshape(-1, 1), yc)
        p_ca = platt.predict_proba(s_ca.reshape(-1, 1))[:, 1]
        p_ho = platt.predict_proba(s_ho.reshape(-1, 1))[:, 1]
        thr = float(np.quantile(p_cv, 0.95))
        pred = p_ho >= thr; tp = int((pred & (yh == 1)).sum())
        rng = np.random.default_rng(7)
        sim = np.array([yh[rng.choice(len(yh), int(pred.sum()), replace=False)].sum()
                        for _ in range(2000)]) if pred.sum() else np.zeros(1)
        model["horizons"][str(h)] = {
            "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
            "platt_coef": float(platt.coef_[0][0]), "platt_intercept": float(platt.intercept_[0]),
            "threshold": thr, "base_rate_calib": float(yc.mean()),
            "calib_cv": calibration_stats(yc, p_cv), "calib_insample_oldmethod": calibration_stats(yc, p_ca),
            "holdout": {"n": len(ho), "pos": int(yh.sum()),
                        "auc": float(roc_auc_score(yh, p_ho)), "pr_auc": float(average_precision_score(yh, p_ho)),
                        "base_rate": float(yh.mean()), "alerts": int(pred.sum()), "hits": tp,
                        "precision": float(tp / max(pred.sum(), 1)), "recall": float(tp / max(yh.sum(), 1)),
                        **calibration_stats(yh, p_ho),
                        "random_precision": float(yh.mean()), "random_recall": float(pred.mean()),
                        "random_hits_mean": float(sim.mean()), "random_p_ge_hits": float((sim >= tp).mean())}}
        hh = model["horizons"][str(h)]["holdout"]
        print(f"[v4 {H_LABEL[h]}] AUC={hh['auc']:.3f} P={hh['precision']:.2f} R={hh['recall']:.2f} "
              f"azar P={hh['random_precision']:.3f} R={hh['random_recall']:.3f} | calib holdout slope={hh['cal_slope']} "
              f"ECE={hh['ece']:.3f} pred={hh['pred_mean']:.3f} obs={hh['obs_rate']:.3f} thr={thr:.3f}")
    model["mag_scale"] = "Mw-equiv Scordilis (v4 recalibrado)"
    MODEL_JSON_V4.write_text(json.dumps(model, indent=1))
    print(f"modelo guardado en {MODEL_JSON_V4}")


def predict_live(n_show=10, fetch_live=False):
    mp = MODEL_JSON_V4 if MODEL_JSON_V4.exists() else (
        MODEL_JSON_V3 if MODEL_JSON_V3.exists() else (
            MODEL_JSON_V2 if MODEL_JSON_V2.exists() else MODEL_JSON))
    if not mp.exists():
        print(f"falta modelo: corre primero --train", file=sys.stderr)
        sys.exit(1)
    model = json.loads(mp.read_text())
    cols = model.get("cols", COLS_V1)
    mean = np.array(model["scaler_mean"])
    scale = np.array(model["scaler_scale"])
    d = load_local()
    msg = f"catalogo local hasta {d.time.max()}"
    if fetch_live:
        try:
            live = fetch_live_usgs(days=90, min_mag=4.0)
            if len(live):
                d = pd.concat([d, live[live.time > d.time.max()]], ignore_index=True)
                d = d.sort_values("time").reset_index(drop=True)
                d["tmin"] = (d.time - pd.Timestamp("1970-01-01")).dt.total_seconds() / 60.0
                msg = f"USGS en vivo: {len(live)} ev M>=4 ultimos 90d; catalogo hasta {d.time.max()}"
            else:
                msg = "USGS en vivo sin eventos nuevos; " + msg
        except Exception as e:
            msg = f"[aviso] USGS en vivo fallo ({e}); " + msg
    tm = d.tmin.to_numpy()
    la, lo, mg, dp = (d[c].to_numpy() for c in ["latitude", "longitude", "mag", "depth"])
    trig = d[(d.mag >= 4.7)].tail(60).reset_index()
    print("=" * 100)
    print("  PRONOSTICO CORTO PLAZO CHILE — P(otro M>=5.0 en 150 km tras cada sismo M>=4.7)")
    print(f"  {msg} | modelo {mp.name} ({model['trained_at'][:10]})")
    print("  OJO: pronostica REPLICAS/escalamiento (clustering). No predice sismos principales.")
    print("-" * 100)
    print(f"  {'hora UTC':<19} {'M':>4} {'lat,lon':>15} {'zona':<24} {'P60min':>7} {'P3h':>7} {'P24h':>7}  alerta")
    print("-" * 100)
    rows = []
    for _, r in trig.iterrows():
        i0 = int(r["index"])
        if cols == COLS_V2:
            f = trigger_features(r, i0, tm, la, lo, mg, dp)
            x = np.array([[f[c] for c in cols]])
        else:  # legacy v1
            lo24 = np.searchsorted(tm, r.tmin - 1440, side="left")
            past = np.arange(lo24, i0)
            dd = hav_km(r.latitude, r.longitude, la[past], lo[past]) if len(past) else np.array([])
            near = past[dd <= 150]
            dt = r.tmin - tm[near]
            m_near = mg[near]
            x = np.array([[np.log1p(np.sum((dt <= 60) & (m_near >= 4.7))), float(r.mag),
                           float(np.log10(1 + dt[m_near >= 4.7].min())) if np.any(m_near >= 4.7) else float(np.log10(1 + 1440)),
                           float(m_near.max()) if len(near) else 4.2]])
        xs = (x - mean) / scale
        probs, flags = {}, []
        for h in HORIZONS_MIN:
            m = model["horizons"][str(h)]
            s = float(np.dot(xs[0], np.array(m["coef"])) + m["intercept"])
            p = float(1 / (1 + math.exp(-(m["platt_coef"] * s + m["platt_intercept"]))))
            probs[h] = p
            if p >= m["threshold"]:
                flags.append(H_LABEL[h])
        rows.append((r, probs, flags))
    for r, probs, flags in (rows[-n_show:] if n_show else []):
        flag = ("ALERTA " + ",".join(flags)) if flags else "—"
        print(f"  {str(r.time)[:19]:<19} {r.mag:4.1f} {r.latitude:6.2f},{r.longitude:7.2f} "
              f"{zone_of(r.latitude):<24} {probs[60] * 100:6.1f}% {probs[180] * 100:6.1f}% {probs[1440] * 100:6.1f}%  {flag}")
    print("-" * 100)
    n_alert = sum(1 for _, _, f in rows if f)
    print(f"  triggers evaluados: {len(rows)} | con alguna alerta: {n_alert}")
    # estado por zona: triggers de los ultimos 7 dias del catalogo
    t_end = d.time.max()
    recent = [(r, p, f) for r, p, f in rows if r.time >= t_end - pd.Timedelta(days=7)]
    print("  ESTADO POR ZONA (7d):")
    if not recent:
        print("    sin triggers M>=4.7 en 7 dias")
    zstat = {}
    for r, p, f in recent:
        z = zone_of(r.latitude)
        zstat.setdefault(z, []).append((p[1440], bool(f)))
    for z, v in sorted(zstat.items(), key=lambda kv: -max(x[0] for x in kv[1])):
        mx = max(x[0] for x in v)
        st = "CALIENTE" if any(x[1] for x in v) else ("ACTIVA" if mx >= 0.08 else "BASAL")
        print(f"    {z:<24} maxP24h={mx * 100:5.1f}%  n={len(v)}  {st}")
    print("=" * 100)
    return rows, d, mp.name


PROSP_LOG = ROOT / "results" / "prospective_log.csv"


def prospective(fetch_live=True):
    """Congela predicciones actuales en results/prospective_log.csv (deduplicado)."""
    rows, d, mname = predict_live(n_show=0, fetch_live=fetch_live)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = pd.DataFrame([{"issued_at": now, "trigger_time": str(r.time), "lat": r.latitude,
                         "lon": r.longitude, "mag": r.mag, "zone": zone_of(r.latitude),
                         "p60": p[60], "p180": p[180], "p1440": p[1440], "model": mname}
                        for r, p, _ in rows])
    if PROSP_LOG.exists():
        old = pd.read_csv(PROSP_LOG)
        seen = set(old.trigger_time.astype(str))
        new = new[~new.trigger_time.astype(str).isin(seen)]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(PROSP_LOG, index=False)
    print(f"prospectivo: +{len(new)} predicciones congeladas -> {PROSP_LOG} (total {len(out)})")


def score_prospective():
    """Evalua entradas del log cuyos horizontes ya maduraron (contra catalogo local)."""
    if not PROSP_LOG.exists():
        print("sin log prospectivo aun")
        return
    log = pd.read_csv(PROSP_LOG, parse_dates=["trigger_time"])
    d = load_local()
    tm = d.tmin.to_numpy()
    la, lo, mg = (d[c].to_numpy() for c in ["latitude", "longitude", "mag"])
    for h, col in [(60, "p60"), (180, "p180"), (1440, "p1440")]:
        ys, ps = [], []
        for _, r in log.iterrows():
            t0 = (r.trigger_time - pd.Timestamp("1970-01-01")).total_seconds() / 60.0
            if (d.time.max() - r.trigger_time).total_seconds() < h * 60:
                continue  # aun no madura
            j = (tm > t0) & (tm <= t0 + h) & (mg >= 5.0)
            y = 0
            if j.sum():
                idx = np.where(j)[0]
                y = int((hav_km(r.lat, r.lon, la[idx], lo[idx]) <= 150).any())
            ys.append(y)
            ps.append(r[col])
        if not ys:
            print(f"  H={h}: sin entradas maduras todavia")
            continue
        ys, ps = np.array(ys), np.array(ps)
        print(f"  H={h}: n={len(ys)} pos={ys.sum()} pred_media={ps.mean():.3f} "
              f"obs={ys.mean():.3f} brier={np.mean((ps - ys) ** 2):.4f}")


def main():
    ap = argparse.ArgumentParser(description="Pronostico sismico de corto plazo (minutos-horas) para Chile")
    ap.add_argument("--train", action="store_true", help="entrena, calibra y backtestea")
    ap.add_argument("--train-v4", action="store_true", help="recalibración honesta v4 (cross-fitted)")
    ap.add_argument("--fetch-live", action="store_true", help="actualiza con USGS en vivo")
    ap.add_argument("--n", type=int, default=10, help="ultimos N triggers a mostrar")
    ap.add_argument("--prospective", action="store_true", help="congela predicciones en el log prospectivo")
    ap.add_argument("--score-prospective", action="store_true", help="evalua entradas maduras del log")
    a = ap.parse_args()
    if a.train_v4:
        train_v4()
    elif a.train:
        train()
    elif a.prospective:
        prospective(fetch_live=True)
    elif a.score_prospective:
        score_prospective()
    else:
        predict_live(n_show=a.n, fetch_live=a.fetch_live)


if __name__ == "__main__":
    main()
