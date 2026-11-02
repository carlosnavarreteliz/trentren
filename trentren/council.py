"""El consejo de Trentren: agentes autónomos que deciden ante cada sismo en Chile, Japón y EE.UU.

TODOS los agentes corren sobre Meta Muse Wonder (api.meta.ai, Responses API). No hay
proveedor alternativo: si Meta no responde, el consejo falla y queda la pre-alerta.

Ante un sismo relevante (real, simulacro o histórico revivido):
  0. Pre-alerta instantánea por reglas + modelo del país, escrita en la BD en < 1 s.
  1. Tres especialistas analizan en paralelo, cada uno con sus herramientas:
       Sismóloga (modelo, secuencia previa, análogos históricos), Vigía de medios
       (prensa e internet), Protección civil (exposición, tsunami).
  2. Debate: cada uno lee a los otros dos y ajusta o defiende su gravedad.
  3. Trentren (coordinador) emite el estado de alerta en es/en/ja; queda en la BD.
"""
import json
import math
import queue
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import db, history, sources
from .config import META_API_KEY, META_BASE_URL, META_MODEL, POLL_SECONDS
from .forecast import Observatorio, ZONE_LABEL, rj_prob
from .regions import PAISES, F, en_pais

NIVELES = ["vigilancia", "atencion", "alerta"]
UMBRAL_CONSEJO = {"chile": 5.0, "japon": 5.5, "eeuu": 5.0}
MAX_CONSEJOS_DIA = 12
IDIOMA = {"es": "español de Chile", "en": "English (US)", "ja": "日本語"}
ORDEN_IDIOMA = {"es": "OBLIGATORIO: escribe TODO (mensajes, informe, réplica) en español.",
                "en": "MANDATORY: write EVERYTHING (messages, report, reply) in English.",
                "ja": "必須：すべて（発言・報告・反論）を日本語で書くこと。スペイン語や英語は使わないこと。"}


def now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- Meta Muse Wonder
class LLMError(Exception):
    pass


META = {"llamadas": 0, "ultimo_modelo": None, "ultima_respuesta": None, "ultimo_ok": None, "errores": 0}


def responses(body, timeout=300):
    """Única puerta a un LLM en todo Trentren: Meta Muse Wonder vía api.meta.ai."""
    if not META_API_KEY:
        raise LLMError("falta META_API_KEY en .env")
    req = urllib.request.Request(META_BASE_URL + "/responses", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + META_API_KEY, "Content-Type": "application/json"})
    err = ""
    for intento in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            META.update(llamadas=META["llamadas"] + 1, ultimo_modelo=out.get("model"), ultima_respuesta=out.get("id"),
                        ultimo_ok=now().isoformat())
            return out
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
            if e.code < 500 and e.code != 429:
                break
        except Exception as e:
            err = str(e)
        time.sleep(4 * (intento + 1))
    META["errores"] += 1
    raise LLMError(err)


def verificar_meta():
    req = urllib.request.Request(META_BASE_URL + "/models", headers={"Authorization": "Bearer " + META_API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            ids = [m["id"] for m in json.loads(r.read()).get("data", [])]
        return {"ok": META_MODEL in ids, "endpoint": META_BASE_URL, "modelo": META_MODEL, "disponibles": ids, **META}
    except Exception as e:
        return {"ok": False, "endpoint": META_BASE_URL, "modelo": META_MODEL, "error": str(e), **META}


def _tool(name, desc, props, req=()):
    return {"type": "function", "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(req)}}


# ---------------------------------------------------------------- exposición
CIUDADES = {  # nombre, lat, lon, población aprox., costera
    "chile": [
        ("Arica", -18.48, -70.31, 222000, 1), ("Iquique", -20.21, -70.15, 191000, 1), ("Alto Hospicio", -20.27, -70.10, 108000, 0),
        ("Tocopilla", -22.09, -70.20, 25000, 1), ("Calama", -22.46, -68.93, 165000, 0), ("San Pedro de Atacama", -22.91, -68.20, 11000, 0),
        ("Mejillones", -23.10, -70.45, 13000, 1), ("Antofagasta", -23.65, -70.40, 361000, 1), ("Taltal", -25.40, -70.48, 13000, 1),
        ("Chañaral", -26.35, -70.62, 13000, 1), ("Caldera", -27.07, -70.82, 17000, 1), ("Copiapó", -27.37, -70.33, 153000, 0),
        ("Vallenar", -28.58, -70.76, 52000, 0), ("La Serena", -29.90, -71.25, 221000, 1), ("Coquimbo", -29.95, -71.34, 227000, 1),
        ("Ovalle", -30.60, -71.20, 111000, 0), ("Illapel", -31.63, -71.17, 31000, 0), ("Los Vilos", -31.91, -71.51, 21000, 1),
        ("La Ligua", -32.45, -71.23, 35000, 0), ("Viña del Mar", -33.02, -71.55, 334000, 1), ("Valparaíso", -33.05, -71.62, 296000, 1),
        ("Gran Santiago", -33.45, -70.66, 6300000, 0), ("San Antonio", -33.59, -71.61, 91000, 1), ("Rancagua", -34.17, -70.74, 241000, 0),
        ("Pichilemu", -34.39, -72.00, 16000, 1), ("Curicó", -34.98, -71.24, 149000, 0), ("Talca", -35.43, -71.66, 220000, 0),
        ("Constitución", -35.33, -72.41, 46000, 1), ("Chillán", -36.61, -72.10, 184000, 0), ("Talcahuano", -36.72, -73.12, 151000, 1),
        ("Concepción", -36.83, -73.05, 223000, 0), ("Lebu", -37.61, -73.65, 25000, 1), ("Los Ángeles", -37.47, -72.35, 202000, 0),
        ("Temuco", -38.74, -72.60, 282000, 0), ("Valdivia", -39.81, -73.25, 166000, 1), ("Osorno", -40.57, -73.13, 161000, 0),
        ("Puerto Montt", -41.47, -72.94, 245000, 1), ("Ancud", -41.87, -73.83, 38000, 1), ("Castro", -42.48, -73.76, 43000, 1),
        ("Coyhaique", -45.57, -72.07, 57000, 0), ("Punta Arenas", -53.16, -70.91, 131000, 1)],
    "japon": [
        ("Sapporo 札幌", 43.06, 141.35, 1970000, 0), ("Kushiro 釧路", 42.98, 144.38, 165000, 1), ("Hakodate 函館", 41.77, 140.73, 250000, 1),
        ("Aomori 青森", 40.82, 140.74, 275000, 1), ("Hachinohe 八戸", 40.51, 141.49, 223000, 1), ("Miyako 宮古", 39.64, 141.95, 50000, 1),
        ("Morioka 盛岡", 39.70, 141.15, 290000, 0), ("Ishinomaki 石巻", 38.43, 141.30, 140000, 1), ("Sendai 仙台", 38.27, 140.87, 1090000, 1),
        ("Fukushima 福島", 37.76, 140.47, 280000, 0), ("Iwaki いわき", 37.05, 140.89, 330000, 1), ("Niigata 新潟", 37.92, 139.04, 780000, 1),
        ("Wajima 輪島", 37.39, 136.90, 25000, 1), ("Kanazawa 金沢", 36.56, 136.66, 460000, 1), ("Mito 水戸", 36.37, 140.47, 270000, 0),
        ("Tokyo 東京", 35.68, 139.69, 14000000, 1), ("Yokohama 横浜", 35.44, 139.64, 3770000, 1), ("Chiba 千葉", 35.61, 140.12, 980000, 1),
        ("Shizuoka 静岡", 34.98, 138.38, 690000, 1), ("Hamamatsu 浜松", 34.71, 137.73, 790000, 1), ("Nagoya 名古屋", 35.18, 136.91, 2330000, 1),
        ("Osaka 大阪", 34.69, 135.50, 2750000, 1), ("Kobe 神戸", 34.69, 135.20, 1520000, 1), ("Kyoto 京都", 35.01, 135.77, 1460000, 0),
        ("Wakayama 和歌山", 34.23, 135.17, 360000, 1), ("Kochi 高知", 33.56, 133.53, 320000, 1), ("Tokushima 徳島", 34.07, 134.56, 250000, 1),
        ("Hiroshima 広島", 34.39, 132.46, 1200000, 1), ("Matsuyama 松山", 33.84, 132.77, 510000, 1), ("Fukuoka 福岡", 33.59, 130.40, 1610000, 1),
        ("Kumamoto 熊本", 32.80, 130.71, 740000, 0), ("Miyazaki 宮崎", 31.91, 131.42, 400000, 1), ("Kagoshima 鹿児島", 31.60, 130.56, 590000, 1),
        ("Naha 那覇", 26.21, 127.68, 320000, 1)],
    "eeuu": [
        ("Seattle", 47.61, -122.33, 750000, 1), ("Tacoma", 47.25, -122.44, 220000, 1), ("Portland", 45.52, -122.68, 650000, 0),
        ("Astoria", 46.19, -123.83, 10000, 1), ("Salem", 44.94, -123.03, 175000, 0), ("Eugene", 44.05, -123.09, 177000, 0),
        ("Newport", 44.64, -124.05, 10000, 1), ("Lincoln City", 44.96, -124.01, 9800, 1), ("Coos Bay", 43.37, -124.22, 16000, 1),
        ("Aberdeen", 46.98, -123.82, 17000, 1), ("Eureka", 40.80, -124.16, 27000, 1), ("Crescent City", 41.76, -124.20, 6700, 1),
        ("Sacramento", 38.58, -121.49, 525000, 0), ("San Francisco", 37.77, -122.42, 810000, 1), ("Oakland", 37.80, -122.27, 430000, 1),
        ("San José", 37.34, -121.89, 970000, 0), ("Fresno", 36.74, -119.79, 545000, 0), ("Santa Barbara", 34.42, -119.70, 88000, 1),
        ("Los Angeles", 34.05, -118.24, 3820000, 1), ("Long Beach", 33.77, -118.19, 450000, 1), ("Riverside", 33.95, -117.40, 315000, 0),
        ("Ridgecrest", 35.62, -117.67, 28000, 0), ("San Diego", 32.72, -117.16, 1380000, 1), ("Las Vegas", 36.17, -115.14, 660000, 0),
        ("Reno", 39.53, -119.81, 270000, 0), ("Salt Lake City", 40.76, -111.89, 200000, 0), ("Anchorage", 61.22, -149.90, 290000, 1),
        ("Kodiak", 57.79, -152.41, 5600, 1), ("Fairbanks", 64.84, -147.72, 32000, 0), ("Juneau", 58.30, -134.42, 32000, 1),
        ("Dutch Harbor", 53.89, -166.54, 4300, 1), ("Honolulu", 21.31, -157.86, 350000, 1), ("Hilo", 19.72, -155.08, 46000, 1)],
}
ROMANO = ["I", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def exposicion(ev):
    out = []
    for n, la, lo, pob, costa in CIUDADES[ev["pais"]]:
        d = float(F.hav_km(ev["lat"], ev["lon"], la, lo))
        mmi = 3.67 + 1.17 * ev["mag"] - 3.19 * math.log10(max(math.hypot(d, ev.get("prof") or 30), 10))
        if mmi >= 3.5:
            out.append({"ciudad": n, "dist_km": round(d), "poblacion": pob, "costera": bool(costa),
                        "intensidad_mercalli": ROMANO[max(1, min(12, int(round(mmi))))], "mmi": round(mmi, 1)})
    out.sort(key=lambda x: -x["mmi"])
    costeras = [c for c in out if c["costera"] and c["dist_km"] < 250]
    somero = (ev.get("prof") or 30) < 60
    cercana = min(out, key=lambda c: c["dist_km"]) if out else None
    interior = bool(cercana and not cercana["costera"] and cercana["dist_km"] < 40)  # epicentro bajo una ciudad interior
    if interior:
        ts = "epicentro en tierra firme: sin condición de tsunami"
    elif ev["mag"] >= 7.0 and somero and costeras:
        ts = "CONDICIÓN DE TSUNAMI: sismo costero mayor y somero. Regla oficial: " + PAISES[ev["pais"]]["tsunami"]
    elif ev["mag"] >= 6.5 and somero and costeras:
        ts = "posible evaluación precautoria de tsunami por la autoridad; mantenerse atento"
    else:
        ts = "sin condición de tsunami por magnitud/profundidad"
    return {"ciudades": out[:12], "poblacion_con_intensidad_VI_o_mas": int(sum(c["poblacion"] for c in out if c["mmi"] >= 5.5)),
            "tsunami": ts, "nota": "Intensidad Mercalli estimada con relación genérica magnitud-distancia; orientativa."}


# ---------------------------------------------------------------- agentes
GRAVEDAD = ("Escala común de gravedad: 1 = menor, apenas sentido; 2 = sentido, sin daños; 3 = fuerte, daños locales o "
            "secuencia activa; 4 = daños importantes, réplicas fuertes probables o riesgo de tsunami; 5 = terremoto destructor.")


def comun(pais):
    P = PAISES[pais]
    return (ORDEN_IDIOMA[P["idioma"]] + " " + f"Eres parte del consejo de TRENTREN, vigía sísmico autónomo (nombre de Tren Tren Vilu, la serpiente de la "
            f"tradición mapuche-chilota que alzó los cerros para salvar a la gente del mar). Vigilas {P['nombre']['es']}. "
            f"Escribe tus mensajes y tu informe en {IDIOMA[P['idioma']]}. Breve y preciso. Antes de usar una herramienta, di "
            f"en una frase qué buscas. Cita cifras exactas de tus herramientas; no inventes datos. {GRAVEDAD} "
            f"Autoridades oficiales que nunca contradices: {', '.join(P['autoridades'])}. "
            "Si el episodio es SIMULACRO, el sismo es ficticio: analízalo igual. Si es HISTÓRICO, revives ese momento: razona "
            "como si fuera esa fecha. Terminas SIEMPRE llamando a tu herramienta de entrega. " + ORDEN_IDIOMA[P["idioma"]])


AGENTES = {
    "sismologa": {
        "nombre": {"es": "Sismóloga", "en": "Seismologist", "ja": "地震学者"},
        "inst": " Tu foco: ¿va a escalar la actividad? ¿Puede anunciar un sismo mayor? Usa `pronostico` siempre, "
                "`sismos_previos` para ver si venía una secuencia y `analogos_historicos` para aprender de sismos parecidos del "
                "pasado en la región. Sé exacta: el modelo pronostica escalamiento y réplicas; nadie predice sismos principales.",
        "tools": ["pronostico", "sismos_previos", "analogos_historicos", "contexto_zona"],
        "entrega": _tool("entregar_informe", "Entrega tu informe al consejo.", {
            "gravedad": {"type": "integer", "description": "1-5"},
            "escalamiento": {"type": "string", "enum": ["bajo", "medio", "alto"]},
            "cifras": {"type": "array", "items": {"type": "string"}, "description": "2-4 cifras clave"},
            "argumento": {"type": "string", "description": "2-3 frases"}}, ["gravedad", "escalamiento", "argumento"]),
    },
    "medios": {
        "nombre": {"es": "Vigía de medios", "en": "Media watch", "ja": "報道ウォッチ"},
        "inst": " Tu foco: ¿cómo lo está tomando la prensa y la gente? Busca noticias con 2 o 3 consultas distintas (lugar, "
                "magnitud, tsunami, autoridad) en el idioma del país. Reporta el tono, los hechos que informan (daños, cortes, "
                "heridos, evacuaciones), las alertas oficiales y los rumores o 'predicciones' falsas. Si no hay cobertura, dilo.",
        "tools": ["buscar_noticias"],
        "entrega": _tool("entregar_informe", "Entrega tu informe al consejo.", {
            "tono": {"type": "string", "enum": ["sin_cobertura", "calma", "preocupacion", "alarma"]},
            "gravedad": {"type": "integer", "description": "1-5, gravedad que refleja la prensa"},
            "hechos": {"type": "array", "items": {"type": "string"}},
            "alertas_oficiales": {"type": "array", "items": {"type": "string"}},
            "rumores": {"type": "array", "items": {"type": "string"}},
            "argumento": {"type": "string", "description": "2-3 frases"}}, ["tono", "gravedad", "argumento"]),
    },
    "proteccion": {
        "nombre": {"es": "Protección civil", "en": "Civil protection", "ja": "防災担当"},
        "inst": " Tu foco: ¿a quién afecta y qué debe hacer la gente? Usa `exposicion` (ciudades, intensidad estimada, "
                "condición de tsunami). Nunca contradigas a la autoridad oficial. Propón 2 a 4 acciones concretas.",
        "tools": ["exposicion", "contexto_zona"],
        "entrega": _tool("entregar_informe", "Entrega tu informe al consejo.", {
            "gravedad": {"type": "integer", "description": "1-5"},
            "riesgo_tsunami": {"type": "string", "enum": ["nulo", "bajo", "medio", "alto"]},
            "poblacion_expuesta": {"type": "string"},
            "acciones": {"type": "array", "items": {"type": "string"}},
            "argumento": {"type": "string", "description": "2-3 frases"}}, ["gravedad", "riesgo_tsunami", "argumento"]),
    },
}
COORD_NOMBRE = {"es": "Trentren", "en": "Trentren", "ja": "トレントレン"}
COORD_INST = (" (El campo razon_decision va en el idioma del país; titular, resumen y acciones en los tres idiomas.)"
              " Eres el COORDINADOR. Lees los informes y el debate del consejo y emites el estado de alerta con `emitir_alerta`; "
              "queda en la base de datos y se publica. Niveles: vigilancia (gravedad 1-2), atencion (3), alerta (4-5); puedes "
              "apartarte si lo justificas. Si los agentes discrepan, di a quién le das la razón y por qué. Escribe el titular, "
              "el resumen (2-4 frases claras, probabilidades en lenguaje simple como '1 de cada 3') y 2-4 acciones en TRES "
              "idiomas: español, inglés y japonés, con el mismo contenido.")
_tri = lambda d: {f"{d}_{l}": {"type": "string"} for l in ("es", "en", "ja")}
_tri_arr = {f"acciones_{l}": {"type": "array", "items": {"type": "string"}} for l in ("es", "en", "ja")}
EMITIR = _tool("emitir_alerta", "Emite el estado de alerta de Trentren (se escribe en la base de datos).", {
    "nivel": {"type": "string", "enum": NIVELES}, "gravedad": {"type": "integer"},
    **_tri("titular"), **_tri("resumen"), **_tri_arr,
    "confianza": {"type": "string", "enum": ["baja", "media", "alta"]},
    "razon_decision": {"type": "string", "description": "por qué, 1-2 frases, citando al consejo"}},
    ["nivel", "gravedad", "titular_es", "titular_en", "titular_ja", "resumen_es", "resumen_en", "resumen_ja",
     "acciones_es", "acciones_en", "acciones_ja", "confianza", "razon_decision"])
POSTURA = _tool("entregar_postura", "Entrega tu postura final tras leer a tus colegas.", {
    "gravedad": {"type": "integer", "description": "1-5, tu gravedad final"},
    "de_acuerdo": {"type": "boolean", "description": "¿coincides en lo esencial con el consejo?"},
    "replica": {"type": "string", "description": "1-3 frases dirigidas a tus colegas"}}, ["gravedad", "de_acuerdo", "replica"])
HERR = {
    "pronostico": _tool("pronostico", "Corre el modelo de escalamiento del país para el sismo en análisis.", {}),
    "sismos_previos": _tool("sismos_previos", "Sismos M≥4 en las horas previas a menos de 300 km.",
                            {"horas": {"type": "integer", "description": "1-336"}}),
    "analogos_historicos": _tool("analogos_historicos", "Sismos parecidos (±0,5 magnitud, <300 km) desde 1990, anteriores al actual, "
                                 "y cuántas réplicas M≥5 tuvieron en 24 h.", {}),
    "contexto_zona": _tool("contexto_zona", "Memoria de largo plazo de la zona: último gran sismo, cuántos desde 1900.", {}),
    "buscar_noticias": _tool("buscar_noticias", "Titulares de Google Noticias del país sobre el sismo.",
                             {"consulta": {"type": "string"}}, ["consulta"]),
    "exposicion": _tool("exposicion", "Ciudades expuestas, intensidad estimada y condición de tsunami.", {}),
}

ESCENARIOS = {
    "chile": {"iquique": {"mag": 7.6, "lat": -20.30, "lon": -70.85, "prof": 28, "lugar": "frente a Iquique, Tarapacá"},
              "valparaiso": {"mag": 6.4, "lat": -33.05, "lon": -71.95, "prof": 32, "lugar": "40 km al O de Valparaíso"},
              "calama": {"mag": 5.9, "lat": -22.40, "lon": -68.60, "prof": 110, "lugar": "20 km al N de Calama"}},
    "japon": {"nankai": {"mag": 7.9, "lat": 33.10, "lon": 135.60, "prof": 20, "lugar": "南海トラフ 紀伊半島沖"},
              "tokyo": {"mag": 6.8, "lat": 35.65, "lon": 139.85, "prof": 45, "lugar": "東京湾北部"},
              "tohoku": {"mag": 6.2, "lat": 38.40, "lon": 142.10, "prof": 40, "lugar": "宮城県沖"}},
    "eeuu": {"cascadia": {"mag": 7.8, "lat": 44.50, "lon": -124.90, "prof": 20, "lugar": "Cascadia, off the Oregon coast"},
             "la": {"mag": 6.7, "lat": 34.05, "lon": -118.10, "prof": 15, "lugar": "Greater Los Angeles"},
             "anchorage": {"mag": 6.4, "lat": 61.40, "lon": -150.00, "prof": 45, "lugar": "near Anchorage, Alaska"}},
}


# ---------------------------------------------------------------- bus
class Bus:
    def __init__(self):
        self.subs, self.lock = [], threading.Lock()

    def sub(self):
        q = queue.Queue(maxsize=300)
        with self.lock:
            self.subs.append(q)
        return q

    def unsub(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    def pub(self, msg):
        with self.lock:
            for q in list(self.subs):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass


# ---------------------------------------------------------------- consejo
class Consejo:
    def __init__(self):
        self.obs = {p: Observatorio(p) for p in PAISES}
        self.hist = {p: history.load(p, self.obs[p].base) for p in PAISES}
        self.bus = Bus()
        self.tasks = queue.Queue()
        self.run_lock = threading.Lock()
        self.vivo = None
        self.snap = {p: {} for p in PAISES}
        self.vistos = []
        self.ultimo_consejo = {}  # (pais, zona) -> (t, mag)
        self.uso = {"in": 0, "out": 0}

    # ---------- ciclo autónomo ----------
    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="trentren").start()

    def _loop(self):
        primera = True
        while True:
            for p in PAISES:
                try:
                    self.snap[p] = self.obs[p].refresh()
                    self.bus.pub({"tipo": "estado", "pais": p})
                    for ev in self._detectar(p, primera):
                        self._ante_sismo(ev)
                    self._cerrar_ventanas(p)
                except Exception:
                    traceback.print_exc()
            primera = False
            t_end = time.time() + POLL_SECONDS
            while time.time() < t_end:
                try:
                    self.tasks.get(timeout=max(1, t_end - time.time()))()
                except queue.Empty:
                    pass
                except Exception:
                    traceback.print_exc()

    def _detectar(self, pais, primera):
        out = []
        for e in self.snap[pais].get("sismos", []):
            if e["mag"] < UMBRAL_CONSEJO[pais] or not en_pais(pais, e["lat"], e["lon"]):
                continue
            t = datetime.fromisoformat(e["t"])
            if any(abs((t - v[0]).total_seconds()) < 180 and abs(e["lat"] - v[1]) < 1 for v in self.vistos):
                continue
            self.vistos.append((t, e["lat"]))
            if not primera and now() - t < timedelta(hours=6):
                out.append({**e, "pais": pais})
        self.vistos = self.vistos[-500:]
        return out

    def _ante_sismo(self, ev):
        """Decide si convoca al consejo o sólo registra la pre-alerta (réplica de una secuencia ya analizada)."""
        p, z = ev["pais"], self.obs[ev["pais"]].zona(ev["lat"], ev["lon"])
        prev = self.ultimo_consejo.get((p, z))
        t = datetime.fromisoformat(ev["t"])
        if prev and t - prev[0] < timedelta(hours=3) and ev["mag"] < prev[1] + 0.5:
            self._instantanea(ev, "real", nota="Réplica de una secuencia que el consejo ya analizó: sólo pre-alerta.")
        elif db.consejos_hoy(p) >= MAX_CONSEJOS_DIA:
            self._instantanea(ev, "real", nota="Límite diario de consejos alcanzado: sólo pre-alerta.")
        else:
            self.ultimo_consejo[(p, z)] = (t, ev["mag"])
            self.consejo(ev, "real")

    def _cerrar_ventanas(self, pais):
        u = db.ultima_real(pais)
        if u and u["nivel"] != "vigilancia" and now() - datetime.fromisoformat(u["t"]) > timedelta(hours=24):
            db.alerta(u["evento_id"], "cierre", "real", "vigilancia", 1, titular="Ventanas cerradas", pais=pais,
                      resumen="Pasaron 24 horas sin escalamiento: vuelta a vigilancia.", autor="Reglas · cierre",
                      i18n={"es": {"titular": "Ventanas cerradas", "resumen": "Pasaron 24 horas sin escalamiento: vuelta a vigilancia."},
                            "en": {"titular": "Windows closed", "resumen": "24 hours without escalation: back to watch."},
                            "ja": {"titular": "監視ウィンドウ終了", "resumen": "24時間、活動の拡大なし。通常監視に戻ります。"}})

    # ---------- simulacro e historia ----------
    def simulacro(self, pais, clave):
        esc = ESCENARIOS[pais][clave]
        ev = {"id": "sim-" + uuid.uuid4().hex[:6], "pais": pais, "fuente": "SIMULACRO",
              "t": (now() - timedelta(minutes=1)).isoformat(), **esc}
        self.tasks.put(lambda: self.consejo(ev, "simulacro"))
        return ev

    def revivir(self, pais, t_iso):
        g = next((x for x in self.hist[pais]["grandes"] if x["t"] == t_iso), None)
        if not g:
            return None
        ev = {"id": f"hist-{pais}-{g['t'][:10]}-{uuid.uuid4().hex[:4]}", "pais": pais, "fuente": "USGS (histórico)",
              "t": g["t"] + "+00:00", "lat": g["lat"], "lon": g["lon"], "prof": g["prof"], "mag": g["mag"],
              "lugar": f"{ZONE_LABEL.get(g['zona'], g['zona'])} · {g['t'][:10]}"}
        self.tasks.put(lambda: self.consejo(ev, "historico"))
        return ev

    # ---------- herramientas ----------
    def _pronostico(self, ev, modo):
        o = self.obs[ev["pais"]]
        t = datetime.fromisoformat(ev["t"])
        if modo != "historico":
            return o.forecast_point(t, ev["lat"], ev["lon"], ev["mag"], ev["prof"])
        fc = o.forecast_point(t, ev["lat"], ev["lon"], ev["mag"], ev["prof"], ahora=t + timedelta(minutes=3))
        tn = pd.Timestamp(t).tz_convert(None)
        H = self.hist[ev["pais"]]
        m = [x for x in H["triggers"] if abs((pd.Timestamp(x[0]) - tn).total_seconds()) < 120 and abs(x[1] - ev["lat"]) < 0.6]
        if m:  # probabilidad walk-forward: sólo con datos anteriores al sismo
            x, prev = m[0], 0.0
            for k, p, hor in (("60min", x[4], 1 / 24), ("3h", x[5], 0.125), ("24h", x[6], 1.0)):
                hz = fc["horizontes"][k]
                pp = max(p, prev, rj_prob(ev["mag"], 0.002, hor, 5.0) if ev["mag"] >= 6 else 0)
                hz.update(p=round(pp, 4), p_v4=p, umbral=round(H["umbral"][k], 4), alerta=pp >= H["umbral"][k],
                          veces_base=round(pp / hz["base"], 1),
                          fuente="walk-forward" + (" + Reasenberg-Jones" if ev["mag"] >= 6 else ""))
                prev = pp
        fc["nota_modo"] = "Modo histórico: probabilidades calculadas sólo con datos anteriores al sismo."
        return fc

    def _ejecutar(self, name, args, ev, modo, fc):
        o = self.obs[ev["pais"]]
        if name == "pronostico":
            return {k: v for k, v in fc.items() if k != "rasgos"} | {
                "lectura": "p = probabilidad de otro sismo M≥5 a <150 km en la ventana; base = tasa normal del país; "
                           "rj.p_mayor_7d = probabilidad de que venga un sismo mayor en 7 días."}
        if name == "sismos_previos":
            h = min(max(int(args.get("horas") or 72), 1), 336)
            t = pd.Timestamp(datetime.fromisoformat(ev["t"])).tz_convert(None)
            c = o.cat
            w = c[(c.time >= t - pd.Timedelta(hours=h)) & (c.time < t - pd.Timedelta(seconds=30)) & (c.mag >= 4.0)]
            if len(w):
                w = w[F.hav_km(ev["lat"], ev["lon"], w.latitude.to_numpy(), w.longitude.to_numpy()) <= 300]
            return {"horas": h, "n": len(w), "m_max": round(float(w.mag.max()), 1) if len(w) else None,
                    "ultimos": [{"t": str(r.time)[:16], "mag": round(float(r.mag), 1), "prof": round(float(r.depth))}
                                for r in w.tail(12).itertuples()]}
        if name == "analogos_historicos":
            return self._analogos(ev)
        if name == "contexto_zona":
            z = o.zona(ev["lat"], ev["lon"])
            return next((x for x in o.memoria if x["zona"] == z), {"zona": z})
        if name == "exposicion":
            return exposicion(ev)
        if name == "buscar_noticias":
            q = args.get("consulta") or PAISES[ev["pais"]]["consulta"]
            loc = PAISES[ev["pais"]]["noticias"]
            try:
                if modo == "historico":
                    d = datetime.fromisoformat(ev["t"]).date()
                    items = sources.search_news(f"{q} after:{d - timedelta(days=1)} before:{d + timedelta(days=4)}",
                                                limit=12, raw=True, loc=loc)
                else:
                    items = sources.search_news(q, days=1, limit=12, loc=loc)
            except Exception as ex:
                return {"error": f"Google Noticias no respondió: {ex}"}
            db.prensa(ev["id"], items)
            p = self.vivo.setdefault("prensa", [])
            p += [i for i in items if all(i["url"] != x["url"] for x in p)]
            return {"consulta": q, "n": len(items), "titulares": [{k: i[k] for k in ("titulo", "medio", "fecha")} for i in items]}
        return {"error": f"herramienta desconocida {name}"}

    def _analogos(self, ev):
        c = self.obs[ev["pais"]].base
        t = pd.Timestamp(datetime.fromisoformat(ev["t"])).tz_convert(None)
        m = ev["mag"]
        w = c[(c.time >= "1990-01-01") & (c.time < t - pd.Timedelta(hours=1)) & (c.mag.between(m - 0.5, m + 0.5))]
        if len(w):
            w = w[F.hav_km(ev["lat"], ev["lon"], w.latitude.to_numpy(), w.longitude.to_numpy()) <= 300]
        tm, la, lo, mg = c.tmin.to_numpy(), c.latitude.to_numpy(), c.longitude.to_numpy(), c.mag.to_numpy()
        out = []
        for r in w.itertuples():
            j = np.where((tm > r.tmin) & (tm <= r.tmin + 1440) & (mg >= 5.0))[0]
            n = int((F.hav_km(r.latitude, r.longitude, la[j], lo[j]) <= 150).sum()) if len(j) else 0
            out.append({"fecha": str(r.time)[:10], "mag": round(float(r.mag), 1), "prof": round(float(r.depth)), "replicas_m5_24h": n})
        k = len(out)
        con = sum(1 for x in out if x["replicas_m5_24h"])
        return {"n": k, "con_replica_m5_24h": con, "fraccion": round(con / k, 2) if k else None, "mas_recientes": out[-8:][::-1]}

    # ---------- motor de un agente (Meta Muse Wonder) ----------
    def _correr(self, ag_id, inst, entrada, tools, entrega, ev, modo, fc, esfuerzo="low", web=False):
        A = self.vivo["agentes"][ag_id]
        tl = [HERR[t] for t in tools] + [entrega] + ([{"type": "web_search"}] if web else [])
        body = {"model": META_MODEL, "instructions": inst, "tools": tl, "reasoning": {"effort": esfuerzo}, "input": entrada}
        for _ in range(7):
            r = responses(body)
            A["modelo"] = r.get("model")
            u = r.get("usage") or {}
            self.uso["in"] += u.get("input_tokens", 0)
            self.uso["out"] += u.get("output_tokens", 0)
            outs, entregado = [], None
            for o in r.get("output", []):
                ty = o.get("type")
                if ty == "message":
                    for c in o.get("content", []):
                        txt = (c.get("text") or "").strip()
                        if txt:
                            A["pasos"].append({"tipo": "dice", "texto": txt})
                elif ty == "web_search_call":
                    q = (o.get("action") or {}).get("query")
                    if q:
                        A["pasos"].append({"tipo": "web", "texto": q})
                elif ty == "function_call":
                    try:
                        args = json.loads(o.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if o["name"] == entrega["name"]:
                        entregado = args
                        continue
                    try:
                        res = self._ejecutar(o["name"], args, ev, modo, fc)
                    except Exception as ex:
                        res = {"error": str(ex)}
                    A["pasos"].append({"tipo": "herramienta", "nombre": o["name"], "args": args, "resumen": _resumir(o["name"], res)})
                    outs.append({"type": "function_call_output", "call_id": o["call_id"],
                                 "output": json.dumps(res, ensure_ascii=False, default=str)[:16000]})
            self._pub()
            if entregado is not None:
                return entregado
            if not outs:
                txt = next((p["texto"] for p in reversed(A["pasos"]) if p["tipo"] == "dice"), "")
                return {"argumento": txt, "gravedad": None}
            body = {"model": META_MODEL, "tools": tl, "reasoning": {"effort": esfuerzo}, "previous_response_id": r["id"], "input": outs}
        return {"argumento": "No alcancé a entregar mi informe.", "gravedad": None}

    def _pub(self):
        self.bus.pub({"tipo": "consejo", "vivo": self.vivo})

    def _instantanea(self, ev, modo, fc=None, nota=None):
        t0 = time.time()
        db.evento(ev, modo)
        fc = fc or self._pronostico(ev, modo)
        nivel, grav = _reglas(ev, fc)
        lat = round(time.time() - t0, 2)
        tit = f"M{ev['mag']:.1f} {ev.get('lugar', '')}"
        res = nota or "Pre-alerta automática por magnitud, profundidad y modelo de escalamiento; el consejo la revisará."
        db.alerta(ev["id"], "instantanea", modo, nivel, grav, fc, titular=tit, resumen=res, autor="Reglas · instantáneo",
                  latencia_s=lat, pais=ev["pais"])
        self.bus.pub({"tipo": "fin", "evento_id": ev["id"]})
        return fc, nivel, grav, lat

    # ---------- episodio completo ----------
    def consejo(self, ev, modo):
        with self.run_lock:
            t_start = time.time()
            pais, P = ev["pais"], PAISES[ev["pais"]]
            fc = self._pronostico(ev, modo)
            fc, nivel0, grav0, lat0 = self._instantanea(ev, modo, fc)
            self.vivo = {"evento": ev, "modo": modo, "pais": pais, "fase": "analisis", "inicio": now().isoformat(),
                         "instantanea": {"nivel": nivel0, "gravedad": grav0, "latencia_s": lat0},
                         "pronostico": {k: v for k, v in fc.items() if k != "rasgos"},
                         "agentes": {k: {"nombre": a["nombre"], "pasos": [], "informe": None, "postura": None, "estado": "pensando"}
                                     for k, a in AGENTES.items()}, "prensa": []}
            self.vivo["agentes"]["coordinador"] = {"nombre": COORD_NOMBRE, "pasos": [], "decision": None, "estado": "esperando"}
            self._pub()
            ctx = self._contexto(ev, modo)
            base = comun(pais)
            try:
                def r1(k):
                    a = AGENTES[k]
                    try:
                        inf = self._correr(k, base + a["inst"], ctx, a["tools"], a["entrega"], ev, modo, fc,
                                           web=(k == "medios" and modo == "real"))
                    except LLMError as ex:
                        inf = {"argumento": f"Meta Muse Wonder no respondió: {ex}", "gravedad": None}
                    self.vivo["agentes"][k].update(informe=inf, estado="listo")
                    db.deliberacion(ev["id"], 1, k, inf.get("argumento", ""), inf.get("gravedad"),
                                    {"informe": inf, "pasos": self.vivo["agentes"][k]["pasos"], "modelo": self.vivo["agentes"][k].get("modelo")})
                    self._pub()
                    return inf
                with ThreadPoolExecutor(3) as ex:
                    informes = dict(zip(AGENTES, ex.map(r1, AGENTES)))
                self.vivo["fase"] = "debate"
                self._pub()

                def r2(k):
                    otros = {AGENTES[j]["nombre"]["es"]: informes[j] for j in AGENTES if j != k}
                    entrada = (ctx + "\n\nTU INFORME: " + json.dumps(informes[k], ensure_ascii=False) +
                               "\n\nINFORMES DE TUS COLEGAS: " + json.dumps(otros, ensure_ascii=False) +
                               "\n\nDebate: ¿coincides? Ajusta o defiende tu gravedad y responde a tus colegas con "
                               "`entregar_postura`. Sé directo; si alguien exagera o subestima, dilo.")
                    self.vivo["agentes"][k]["estado"] = "debatiendo"
                    self._pub()
                    try:
                        pos = self._correr(k, base + AGENTES[k]["inst"], entrada, [], POSTURA, ev, modo, fc)
                    except LLMError as ex2:
                        pos = {"replica": f"(sin respuesta de Meta: {ex2})", "gravedad": informes[k].get("gravedad"), "de_acuerdo": True}
                    self.vivo["agentes"][k].update(postura=pos, estado="listo")
                    db.deliberacion(ev["id"], 2, k, pos.get("replica", ""), pos.get("gravedad"), {"postura": pos})
                    self._pub()
                    return pos
                with ThreadPoolExecutor(3) as ex:
                    posturas = dict(zip(AGENTES, ex.map(r2, AGENTES)))
                self.vivo["fase"] = "decision"
                self.vivo["agentes"]["coordinador"]["estado"] = "decidiendo"
                self._pub()
                acta = {AGENTES[k]["nombre"]["es"]: {"informe": informes[k], "postura_final": posturas[k]} for k in AGENTES}
                entrada = (ctx + "\n\nPRE-ALERTA INSTANTÁNEA: " + json.dumps({"nivel": nivel0, "gravedad": grav0}) +
                           "\n\nACTA DEL CONSEJO: " + json.dumps(acta, ensure_ascii=False) + "\n\nEmite el estado con `emitir_alerta`.")
                dec = self._correr("coordinador", base + COORD_INST, entrada, ["pronostico"], EMITIR, ev, modo, fc, esfuerzo="medium")
            except Exception as ex:
                traceback.print_exc()
                dec = {"nivel": nivel0, "gravedad": grav0, "titular_es": f"M{ev['mag']:.1f} {ev.get('lugar', '')}",
                       "resumen_es": f"El consejo no pudo completar la deliberación ({ex}). Se mantiene la pre-alerta.",
                       "confianza": "baja", "razon_decision": "falla técnica"}
            if dec.get("nivel") not in NIVELES:
                dec["nivel"] = nivel0
            i18n = {l: {"titular": dec.get(f"titular_{l}") or dec.get("titular_es", ""),
                        "resumen": dec.get(f"resumen_{l}") or dec.get("resumen_es", ""),
                        "acciones": dec.get(f"acciones_{l}") or dec.get("acciones_es") or []} for l in ("es", "en", "ja")}
            lat = round(time.time() - t_start, 1)
            modelo = self.vivo["agentes"]["coordinador"].get("modelo")
            aid = db.alerta(ev["id"], "consejo", modo, dec["nivel"], dec.get("gravedad") or grav0, fc,
                            titular=i18n["es"]["titular"], resumen=i18n["es"]["resumen"], acciones=i18n["es"]["acciones"],
                            confianza=dec.get("confianza", ""), autor="Trentren · consejo", latencia_s=lat, pais=pais,
                            i18n=i18n, modelo_llm=modelo)
            final = {"nivel": dec["nivel"], "gravedad": dec.get("gravedad"), "i18n": i18n, "confianza": dec.get("confianza"),
                     "razon_decision": dec.get("razon_decision"), "registro_id": aid, "latencia_s": lat, "modelo": modelo}
            if modo == "historico":
                final["realidad"] = self._realidad(ev)
            db.deliberacion(ev["id"], 3, "coordinador", dec.get("razon_decision", ""), dec.get("gravedad"), {"decision": final})
            self.vivo["agentes"]["coordinador"].update(decision=final, estado="listo")
            self.vivo["fase"] = "fin"
            self._pub()
            self.vivo = None
            self.bus.pub({"tipo": "fin", "evento_id": ev["id"]})

    def _realidad(self, ev):
        c = self.obs[ev["pais"]].base
        t = pd.Timestamp(datetime.fromisoformat(ev["t"])).tz_convert(None)
        out = {}
        for k, m in (("60min", 60), ("3h", 180), ("24h", 1440), ("7d", 10080)):
            w = c[(c.time > t + pd.Timedelta(seconds=30)) & (c.time <= t + pd.Timedelta(minutes=m)) & (c.mag >= 5.0)]
            if len(w):
                w = w[F.hav_km(ev["lat"], ev["lon"], w.latitude.to_numpy(), w.longitude.to_numpy()) <= 150]
            out[k] = {"n_m5": len(w), "mayor": round(float(w.mag.max()), 1) if len(w) else None}
        return out

    def _contexto(self, ev, modo):
        P = PAISES[ev["pais"]]
        tz = ZoneInfo(P["tz"])
        t = datetime.fromisoformat(ev["t"])
        ahora = t + timedelta(minutes=3) if modo == "historico" else now()
        return json.dumps({
            "episodio": {"simulacro": "SIMULACRO (sismo ficticio)", "historico": "HISTÓRICO (reviviendo un sismo real)",
                         "real": "REAL (en vivo)"}[modo],
            "pais": P["nombre"]["es"],
            "sismo": {"magnitud": ev["mag"], "lugar": ev.get("lugar"), "lat": ev["lat"], "lon": ev["lon"],
                      "profundidad_km": ev.get("prof"), "hora_local": t.astimezone(tz).strftime("%Y-%m-%d %H:%M"),
                      "utc": ev["t"][:19], "fuente": ev.get("fuente"), "zona": self.obs[ev["pais"]].zona(ev["lat"], ev["lon"])},
            "ahora_local": ahora.astimezone(tz).strftime("%Y-%m-%d %H:%M"),
            "regla_tsunami_oficial": P["tsunami"],
            "IDIOMA_OBLIGATORIO": ORDEN_IDIOMA[P["idioma"]],
        }, ensure_ascii=False)

    # ---------- estado público ----------
    def publico(self, pais):
        u = db.ultima_real(pais)
        vigente = u if u and now() - datetime.fromisoformat(u["t"]) < timedelta(hours=24) and u["etapa"] == "consejo" else None
        P = PAISES[pais]
        return {"pais": pais, "info": {"nombre": P["nombre"], "tz": P["tz"], "vista": P["vista"], "links": P["links"],
                                       "tsunami": P["tsunami"], "escenarios": {k: v for k, v in ESCENARIOS[pais].items()}},
                "vivo": self.vivo, "vigente": vigente, "nivel": (vigente or {}).get("nivel", "vigilancia"),
                "registro": db.registro(30, pais), "consejos": db.consejos_recientes(6, pais),
                "modelo_llm": META_MODEL, "meta": {k: META[k] for k in ("llamadas", "ultimo_modelo", "ultimo_ok")},
                "uso": self.uso, "cola": self.tasks.qsize(),
                "historia": {k: v for k, v in self.hist[pais].items() if k != "triggers"}, **self.snap[pais]}

    def momento(self, pais, t):
        return history.momento(pais, self.obs[pais].base, self.hist[pais], t)


def _reglas(ev, fc):
    m, z = ev["mag"], ev.get("prof") or 30
    al = any(h["alerta"] for h in fc["horizontes"].values())
    if m >= 7.0 or (m >= 6.5 and z < 70):
        return "alerta", 5 if m >= 7.5 else 4
    if m >= 6.0 or (al and m >= 5.5):
        return "atencion", 3
    if al:
        return "atencion", 2
    return "vigilancia", 2 if m >= 5.0 else 1


def _resumir(name, res):
    if "error" in res:
        return "error: " + str(res["error"])[:140]
    if name == "pronostico":
        h = res["horizontes"]
        return "P(M≥5) 1h {:.0%} · 3h {:.0%} · 24h {:.0%} · mayor 7d {:.1%}".format(
            h["60min"]["p"], h["3h"]["p"], h["24h"]["p"], res["rj"]["p_mayor_7d"])
    if name == "sismos_previos":
        return f"{res['n']} sismos M≥4 en {res['horas']} h" + (f" · máx M{res['m_max']}" if res["m_max"] else "")
    if name == "analogos_historicos":
        return f"{res['n']} análogos desde 1990 · {res['con_replica_m5_24h']} con réplica M≥5 en 24 h" if res["n"] else "sin análogos"
    if name == "contexto_zona":
        return f"{res.get('nombre', res.get('zona'))}: último gran sismo {res.get('ultimo_gran') or '—'}"
    if name == "exposicion":
        c = res["ciudades"][:3]
        return (", ".join(f"{x['ciudad']} {x['intensidad_mercalli']}" for x in c) or "sin ciudades expuestas") + " · " + res["tsunami"][:60]
    if name == "buscar_noticias":
        return f"«{res['consulta']}» → {res['n']} titulares" + (f": {res['titulares'][0]['titulo'][:80]}" if res["n"] else "")
    return "hecho"
