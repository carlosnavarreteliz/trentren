"""El ojo de Trentren: catálogo vivo + modelo v4 de escalamiento + Reasenberg-Jones, por país.

Cada país usa SU modelo, entrenado con sus propios sismos pasados (regions.train).
No predice sismos principales: estima la probabilidad de que la actividad escale.
"""
import json
import math
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import EVENT_MW
from . import sources
from .regions import PAISES, F, load, zona

H = F.HORIZONS_MIN  # [60, 180, 1440]
EPOCH = pd.Timestamp("1970-01-01")
RJ = {"a": -1.67, "b": 0.91, "p": 1.08, "c": 0.05}  # Reasenberg & Jones (1989), genéricos
ZONE_LABEL = {
    "Arica-Parinacota": "Arica y Parinacota", "Iquique-Loa": "Tarapacá · Loa",
    "Antofagasta-Calama": "Antofagasta", "Atacama-Copiapo": "Atacama",
    "Coquimbo-Illapel": "Coquimbo", "Valparaiso-RM": "Valparaíso · Metropolitana",
    "OHiggins-Maule-Biobio": "O'Higgins · Maule · Biobío",
    "Araucania-Valdivia-Chiloe": "Araucanía · Los Lagos", "Austral": "Aysén · Magallanes",
}


def _now():
    return datetime.now(timezone.utc)


def rj_prob(mag, t_elapsed_days, horizon_days, m_min):
    a, b, p, c = RJ["a"], RJ["b"], RJ["p"], RJ["c"]
    t1 = max(t_elapsed_days, 0.0)
    t2 = t1 + horizon_days
    n = 10 ** (a + b * (mag - m_min)) * ((t2 + c) ** (1 - p) - (t1 + c) ** (1 - p)) / (1 - p)
    return 1 - math.exp(-n)


class Observatorio:
    def __init__(self, pais):
        self.pais = pais
        self.P = PAISES[pais]
        self.lock = threading.Lock()
        self.model = json.loads(self.P["modelo"].read_text())
        self.cols = self.model.get("cols", F.COLS_V2)
        self.mean = np.array(self.model["scaler_mean"])
        self.scale = np.array(self.model["scaler_scale"])
        self.base = load(pais)
        self.cat = self.base
        self.display, self.triggers = [], []
        self.status = {"csn": None, "usgs": None, "errores": []}
        self.memoria = self._memoria_tierra()

    def zona(self, lat, lon):
        return zona(self.pais, lat, lon)

    # ---------- percepción ----------
    def refresh(self):
        errs, cat, usgs_ok, csn_ok = [], self.base, None, None
        try:
            live = sources.fetch_usgs_cat(self.P["cajas"], days=30, min_mag=4.0)
            if len(live):
                live["mag"] = [F.to_mw(m, t) for m, t in zip(live.mag, live.magType)]
                cat = pd.concat([self.base, live[live.time > self.base.time.max()]], ignore_index=True)
                cat = cat.sort_values("time").reset_index(drop=True)
                cat["tmin"] = (cat.time - EPOCH).dt.total_seconds() / 60.0
            usgs_ok = _now().isoformat()
        except Exception as e:
            errs.append(f"USGS catálogo: {e}")
        disp_us, disp_csn = [], []
        try:
            disp_us = sources.fetch_usgs(self.P["cajas"], days=7, min_mag=2.5)
        except Exception as e:
            errs.append(f"USGS 7d: {e}")
        if self.pais == "chile":
            try:
                disp_csn = sources.fetch_csn()
                csn_ok = _now().isoformat()
            except Exception as e:
                errs.append(f"CSN: {e}")
        display = self._merge(disp_csn, disp_us)
        with self.lock:
            self.cat, self.display = cat, display
            self.status = {"csn": csn_ok or self.status.get("csn"), "usgs": usgs_ok or self.status.get("usgs"),
                           "errores": errs, "catalogo_hasta": str(cat.time.max())[:19]}
            self.triggers = self._triggers()
        return self.snapshot()

    @staticmethod
    def _merge(csn, usgs):
        out = list(csn)
        for u in usgs:
            tu = datetime.fromisoformat(u["t"])
            dup = next((c for c in csn if abs((datetime.fromisoformat(c["t"]) - tu).total_seconds()) < 90
                        and abs(c["lat"] - u["lat"]) < 1.0 and abs(c["lon"] - u["lon"]) < 1.0), None)
            if dup:
                dup["tsunami"] = u.get("tsunami", 0)
            else:
                out.append(u)
        out.sort(key=lambda e: e["t"], reverse=True)
        return out

    # ---------- modelo ----------
    def forecast_point(self, t, lat, lon, mag, depth, cat=None, ahora=None):
        cat = self.cat if cat is None else cat
        t = pd.Timestamp(t)
        t = t.tz_convert(None) if t.tzinfo else t
        tmin = (t - EPOCH).total_seconds() / 60.0
        tm = cat.tmin.to_numpy()
        la, lo, mg, dp = (cat[c].to_numpy() for c in ["latitude", "longitude", "mag", "depth"])
        i0 = int(np.searchsorted(tm, tmin - 0.5, side="left"))
        r = pd.Series({"tmin": tmin, "latitude": lat, "longitude": lon, "mag": mag, "depth": depth})
        f = F.trigger_features(r, i0, tm, la, lo, mg, dp)
        xs = (np.array([[f[c] for c in self.cols]]) - self.mean) / self.scale
        now = pd.Timestamp(ahora or _now())
        elapsed_min = max(0.0, ((now.tz_convert(None) if now.tzinfo else now) - t).total_seconds() / 60.0)
        ed = elapsed_min / 1440
        out = {"horizontes": {}, "rasgos": {k: round(v, 3) for k, v in f.items()}, "modelo_pais": self.pais}
        prev = 0.0
        for h in H:
            m = self.model["horizons"][str(h)]
            s = float(xs[0] @ np.array(m["coef"]) + m["intercept"])
            p4 = 1 / (1 + math.exp(-(m["platt_coef"] * s + m["platt_intercept"])))
            # P acumulada no baja al ampliar la ventana; con M>=6 el logit extrapola: máximo con R-J.
            p = max(p4, prev)
            if mag >= 6.0:
                p = max(p, rj_prob(mag, ed, h / 1440, 5.0))
            prev = p
            base = m["base_rate_calib"]
            out["horizontes"][F.H_LABEL[h]] = {
                "p": round(p, 4), "p_v4": round(p4, 4), "base": round(base, 4), "veces_base": round(p / base, 1),
                "umbral": round(m["threshold"], 4), "alerta": p >= m["threshold"],
                "abierta": elapsed_min < h, "minutos_restantes": max(0, int(h - elapsed_min)),
                "fuente": "v4 + Reasenberg-Jones" if mag >= 6.0 else "v4 logit"}
        out["rj"] = {"p_mayor_7d": round(rj_prob(mag, ed, 7, mag + 0.1), 4),
                     "p_m5_7d": round(rj_prob(mag, ed, 7, 5.0), 4) if mag >= 5.0 else None}
        return out

    def _triggers(self):
        cat = self.cat
        trig = cat[(cat.mag >= EVENT_MW) & (cat.time >= cat.time.max() - pd.Timedelta(days=7))]
        rows = []
        for _, r in trig.iterrows():
            fc = self.forecast_point(r.time, r.latitude, r.longitude, r.mag, r.depth, cat=cat)
            fc.pop("rasgos", None)
            rows.append({"id": f"cat-{int(r.tmin)}", "t": r.time.tz_localize("UTC").isoformat(),
                         "lat": round(float(r.latitude), 3), "lon": round(float(r.longitude), 3),
                         "mag": round(float(r.mag), 1), "prof": round(float(r.depth), 0),
                         "zona": self.zona(r.latitude, r.longitude), **fc})
        rows.sort(key=lambda x: x["t"], reverse=True)
        return rows

    def _memoria_tierra(self):
        d = self.base[self.base.time >= "1900-01-01"]
        now = pd.Timestamp(_now()).tz_convert(None)
        segs = {}
        if self.pais == "chile":
            from .config import ROOT
            for (n, *_), s in zip(F.ZONES, json.loads((ROOT / "data/raw/chile_tectonic_segments.json").read_text())):
                segs[n] = s
        out = []
        for name, a, b, c, e in self.P["zonas"]:
            z = d[d.latitude.between(a, b) & d.longitude.between(c, e)]
            big = z[z.mag >= 7.5].sort_values("time")
            last = big.iloc[-1] if len(big) else None
            s = segs.get(name, {})
            out.append({"zona": name, "nombre": ZONE_LABEL.get(name, name), "lat_min": a, "lat_max": b, "lon_min": c, "lon_max": e,
                        "ultimo_gran": f"{last.time.year} · M{last.mag:.1f}" if last is not None else None,
                        "anios_desde_gran": round((now - last.time).days / 365.25, 1) if last is not None else None,
                        "grandes_desde_1900": int(len(big)),
                        "acoplamiento": s.get("geodetic_locking_mean"),
                        "ultima_ruptura_gigante": s.get("last_giant_rupture_year"), "mag_ruptura_gigante": s.get("last_giant_mag")})
        return out

    # ---------- estado ----------
    def zonas(self):
        now = _now()
        out = []
        for zm in self.memoria:
            ev = [e for e in self.display if self.zona(e["lat"], e["lon"]) == zm["zona"]]
            ev24 = [e for e in ev if (now - datetime.fromisoformat(e["t"])) < timedelta(days=1)]
            tz = [t for t in self.triggers if t["zona"] == zm["zona"]]
            open_p = [t["horizontes"]["24h"]["p"] for t in tz if t["horizontes"]["24h"]["abierta"]]
            out.append({**zm, "sismos_7d": len(ev), "sismos_24h": len(ev24),
                        "mayor_7d": max((e["mag"] for e in ev), default=None),
                        "p24_abierta": max(open_p) if open_p else None,
                        "alerta_modelo": any(h["alerta"] and h["abierta"] for t in tz for h in t["horizontes"].values())})
        return out

    def metricas(self):
        m = self.model
        return {"modelo": self.P["modelo"].name, "entrenado": m.get("trained_at", "")[:10],
                "n_train": m.get("n_train"), "n_calib": m.get("n_calib"), "n_holdout": m.get("n_holdout"),
                "periodo_train": m.get("train_period", "1990-2020"),
                "holdout": {F.H_LABEL[h]: {k: m["horizons"][str(h)]["holdout"].get(k) for k in
                                           ("n", "pos", "auc", "precision", "recall", "base_rate", "alerts", "hits")} for h in H}}

    def snapshot(self):
        with self.lock:
            return {"sismos": self.display[:500], "triggers": self.triggers[:20], "zonas": self.zonas(),
                    "fuentes": self.status, "metricas": self.metricas()}
