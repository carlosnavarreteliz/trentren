"""Máquina del tiempo: cómo se habría visto Trentren cada año desde 2014, por país.

Reconstrucción walk-forward (misma lógica que dashboard/generate.py:replay_data):
logit entrenado sólo con sismos anteriores a 2014; la calibración de cada año Y
se ajusta sólo con años < Y. Ninguna probabilidad mira el futuro.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import STATE
from .regions import F, PAISES, zona

HZ = [("60min", 60), ("3h", 180), ("24h", 1440)]


def _cache(pais):
    return STATE / f"historia_{pais}.json"


def replay(df):
    cols = F.COLS_V2
    tr = df[df.time < "2014-01-01"].reset_index(drop=True)
    nva = int(0.15 * len(tr))
    trf, trv = tr.iloc[:-nva], tr.iloc[-nva:]
    te = df[df.time >= "2014-01-01"].reset_index(drop=True)
    sc = StandardScaler().fit(trf[cols].to_numpy())
    P, THR = {}, {}
    yrs = te.time.dt.year.to_numpy()
    for k, h in HZ:
        clf = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(trf[cols].to_numpy()), trf[f"y_{h}"].to_numpy())
        sv = clf.decision_function(sc.transform(trv[cols].to_numpy()))
        yv = trv[f"y_{h}"].to_numpy()
        THR[k] = float(np.quantile(LogisticRegression(C=1e6).fit(sv.reshape(-1, 1), yv).predict_proba(sv.reshape(-1, 1))[:, 1], 0.95))
        ste = clf.decision_function(sc.transform(te[cols].to_numpy()))
        out = np.zeros(len(te))
        s_hist, y_hist = list(sv), list(yv)
        for Y in sorted(set(yrs)):
            mk = yrs == Y
            out[mk] = LogisticRegression(C=1e6).fit(np.array(s_hist).reshape(-1, 1), np.array(y_hist)).predict_proba(ste[mk].reshape(-1, 1))[:, 1]
            s_hist += list(ste[mk])
            y_hist += list(te.loc[mk, f"y_{h}"].to_numpy())
        P[k] = out
    return te, P, THR


def _met(g, P_col, y_col, thr):
    al = g[P_col] >= thr
    yy = g[y_col] == 1
    base = float(yy.mean()) if len(g) else 0.0
    prec = float((al & yy).sum() / max(al.sum(), 1))
    return {"escaladas": int(yy.sum()), "alertas": int(al.sum()), "aciertos": int((al & yy).sum()),
            "precision": round(prec, 3), "base": round(base, 4),
            "veces_azar": round(prec / base, 1) if base > 0 and al.sum() else None,
            "recall": round(float((al & yy).sum() / max(yy.sum(), 1)), 3)}


def build(pais, cat):
    df = F.build_dataset(cat)
    te, P, thr = replay(df)
    d = pd.DataFrame({"t": te.time.astype(str).str[:19], "lat": te.lat.round(2), "lon": te.lon.round(2), "mag": te.x2.round(1),
                      "p60min": P["60min"].round(3), "p3h": P["3h"].round(3), "p24h": P["24h"].round(3),
                      "y60min": te.y_60, "y3h": te.y_180, "y24h": te.y_1440})
    d["zona"] = [zona(pais, a, b) for a, b in zip(d.lat, d.lon)]
    d["anio"] = d.t.str[:4].astype(int)
    anios = [{"anio": int(y), "n": len(g), **{k: _met(g, f"p{k}", f"y{k}", thr[k]) for k, _ in HZ}} for y, g in d.groupby("anio")]
    total = {k: {"n": len(d), **_met(d, f"p{k}", f"y{k}", thr[k])} for k, _ in HZ}

    big = cat[(cat.time >= "2014-01-01") & (cat.mag >= 6.8)].sort_values("mag", ascending=False).head(14).sort_values("time")
    tm, la, lo, mg = cat.tmin.to_numpy(), cat.latitude.to_numpy(), cat.longitude.to_numpy(), cat.mag.to_numpy()
    dt = pd.to_datetime(d.t)
    grandes = []
    for _, b in big.iterrows():
        bt = b.time
        near = d[(dt >= bt - pd.Timedelta(days=14)) & (dt < bt - pd.Timedelta(minutes=1))]
        if len(near):
            near = near[F.hav_km(b.latitude, b.longitude, near.lat.to_numpy(), near.lon.to_numpy()) <= 150]
        own = d[(abs((dt - bt).dt.total_seconds()) < 120)]
        idx = np.where((tm > b.tmin) & (tm <= b.tmin + 1440) & (mg >= 5.0))[0]
        aft = idx[F.hav_km(b.latitude, b.longitude, la[idx], lo[idx]) <= 150] if len(idx) else idx
        grandes.append({"t": str(bt)[:19], "mag": round(float(b.mag), 1), "lat": round(float(b.latitude), 2),
                        "lon": round(float(b.longitude), 2), "prof": round(float(b.depth)), "zona": zona(pais, b.latitude, b.longitude),
                        "previos_14d": int(len(near)), "previos_alerta": int((near.p24h >= thr["24h"]).sum()) if len(near) else 0,
                        "previo_max_p24": round(float(near.p24h.max()), 3) if len(near) else None,
                        "modelo": {k: float(own.iloc[0][f"p{k}"]) for k, _ in HZ} if len(own) else None,
                        "replicas_m5_24h": int(len(aft)), "mayor_replica_24h": round(float(mg[aft].max()), 1) if len(aft) else None})
    trig = d[["t", "lat", "lon", "mag", "p60min", "p3h", "p24h", "y60min", "y3h", "y24h", "zona"]].values.tolist()
    out = {"pais": pais, "umbral": thr, "anios": anios, "total": total, "grandes": grandes, "triggers": trig,
           "n_train_replay": int((df.time < "2014-01-01").sum())}
    _cache(pais).write_text(json.dumps(out, ensure_ascii=False, default=float))
    return out


def load(pais, cat):
    f = _cache(pais)
    if f.exists():
        return json.loads(f.read_text())
    return build(pais, cat)


def momento(pais, cat, hist, t_iso, dias=7):
    """El territorio en el instante t: sismos de los 7 días previos, y lo que vino en las 24 h siguientes."""
    t = pd.Timestamp(t_iso)
    t = t.tz_convert(None) if t.tzinfo else t
    def ev(dd):
        return [{"t": r.time.isoformat() + "+00:00", "lat": round(float(r.latitude), 3), "lon": round(float(r.longitude), 3),
                 "mag": round(float(r.mag), 1), "prof": round(float(r.depth)), "fuente": "USGS",
                 "lugar": zona(pais, r.latitude, r.longitude), "tipo_mag": "Mw-eq"} for r in dd.itertuples()]
    win = cat[(cat.time > t - pd.Timedelta(days=dias)) & (cat.time <= t + pd.Timedelta(seconds=1)) & (cat.mag >= 4.0)]
    aft = cat[(cat.time > t + pd.Timedelta(seconds=1)) & (cat.time <= t + pd.Timedelta(days=1)) & (cat.mag >= 4.0)]
    tr = [x for x in hist["triggers"] if t - pd.Timedelta(days=dias) < pd.Timestamp(x[0]) <= t + pd.Timedelta(seconds=1)]
    return {"t": t.isoformat(), "sismos": ev(win)[::-1], "despues_24h": ev(aft), "triggers": tr[::-1][:10], "umbral": hist["umbral"]}
