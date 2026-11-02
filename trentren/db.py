"""Registro de Trentren en SQLite: eventos, alertas, deliberaciones y prensa."""
import json
import sqlite3
import threading
from datetime import datetime, timezone

from .config import STATE

DB = STATE / "trentren.db"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS eventos(
  id TEXT PRIMARY KEY, t TEXT, lat REAL, lon REAL, prof REAL, mag REAL, lugar TEXT,
  fuente TEXT, modo TEXT, creado TEXT, pais TEXT);
CREATE TABLE IF NOT EXISTS alertas(
  id INTEGER PRIMARY KEY AUTOINCREMENT, evento_id TEXT, t TEXT, etapa TEXT, modo TEXT,
  nivel TEXT, gravedad INTEGER, p60 REAL, p3h REAL, p24h REAL, p_mayor REAL,
  titular TEXT, resumen TEXT, acciones TEXT, confianza TEXT, autor TEXT, latencia_s REAL,
  pais TEXT, i18n TEXT, modelo_llm TEXT);
CREATE TABLE IF NOT EXISTS deliberaciones(
  id INTEGER PRIMARY KEY AUTOINCREMENT, evento_id TEXT, ronda INTEGER, agente TEXT, t TEXT,
  texto TEXT, gravedad INTEGER, datos TEXT);
CREATE TABLE IF NOT EXISTS prensa(
  id INTEGER PRIMARY KEY AUTOINCREMENT, evento_id TEXT, titulo TEXT, medio TEXT, fecha TEXT, url TEXT);
CREATE INDEX IF NOT EXISTS ix_al_ev ON alertas(evento_id);
CREATE INDEX IF NOT EXISTS ix_de_ev ON deliberaciones(evento_id);
"""


def _con():
    c = sqlite3.connect(DB, timeout=10, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


with _con() as _c:
    _c.executescript(SCHEMA)


def now():
    return datetime.now(timezone.utc).isoformat()


def evento(e, modo):
    with _lock, _con() as c:
        c.execute("INSERT OR REPLACE INTO eventos VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (e["id"], e["t"], e["lat"], e["lon"], e.get("prof"), e["mag"], e.get("lugar", ""),
                   e.get("fuente", ""), modo, now(), e.get("pais", "chile")))


def alerta(evento_id, etapa, modo, nivel, gravedad, fc=None, titular="", resumen="", acciones=(),
           confianza="", autor="", latencia_s=None, pais="chile", i18n=None, modelo_llm=None):
    h = (fc or {}).get("horizontes", {})
    g = lambda k: h.get(k, {}).get("p")
    with _lock, _con() as c:
        cur = c.execute(
            "INSERT INTO alertas(evento_id,t,etapa,modo,nivel,gravedad,p60,p3h,p24h,p_mayor,titular,resumen,acciones,"
            "confianza,autor,latencia_s,pais,i18n,modelo_llm) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (evento_id, now(), etapa, modo, nivel, gravedad, g("60min"), g("3h"), g("24h"),
             (fc or {}).get("rj", {}).get("p_mayor_7d"), titular, resumen, json.dumps(list(acciones), ensure_ascii=False),
             confianza, autor, latencia_s, pais, json.dumps(i18n or {}, ensure_ascii=False), modelo_llm))
        return cur.lastrowid


def deliberacion(evento_id, ronda, agente, texto, gravedad=None, datos=None):
    with _lock, _con() as c:
        c.execute("INSERT INTO deliberaciones(evento_id,ronda,agente,t,texto,gravedad,datos) VALUES (?,?,?,?,?,?,?)",
                  (evento_id, ronda, agente, now(), texto, gravedad, json.dumps(datos or {}, ensure_ascii=False)))


def prensa(evento_id, items):
    with _lock, _con() as c:
        c.executemany("INSERT INTO prensa(evento_id,titulo,medio,fecha,url) VALUES (?,?,?,?,?)",
                      [(evento_id, i.get("titulo"), i.get("medio"), i.get("fecha"), i.get("url")) for i in items])


def _rows(q, args=()):
    with _con() as c:
        return [dict(r) for r in c.execute(q, args)]


def _dec(r):
    r["acciones"] = json.loads(r.get("acciones") or "[]")
    r["i18n"] = json.loads(r.get("i18n") or "{}")


def registro(limit=40, pais=None):
    w = "WHERE a.pais=?" if pais else ""
    rows = _rows("SELECT a.*, e.mag, e.lugar, e.t AS t_evento FROM alertas a LEFT JOIN eventos e ON e.id=a.evento_id "
                 f"{w} ORDER BY a.id DESC LIMIT ?", ((pais, limit) if pais else (limit,)))
    for r in rows:
        _dec(r)
    return rows


def ultima_real(pais):
    r = _rows("SELECT a.*, e.mag, e.lugar FROM alertas a LEFT JOIN eventos e ON e.id=a.evento_id "
              "WHERE a.modo='real' AND a.pais=? AND a.etapa IN ('consejo','cierre') ORDER BY a.id DESC LIMIT 1", (pais,))
    if r:
        _dec(r[0])
    return r[0] if r else None


def consejo(evento_id):
    ev = _rows("SELECT * FROM eventos WHERE id=?", (evento_id,))
    if not ev:
        return None
    de = _rows("SELECT * FROM deliberaciones WHERE evento_id=? ORDER BY id", (evento_id,))
    for d in de:
        d["datos"] = json.loads(d["datos"] or "{}")
    al = _rows("SELECT * FROM alertas WHERE evento_id=? ORDER BY id", (evento_id,))
    for a in al:
        _dec(a)
    pr = _rows("SELECT titulo,medio,fecha,url FROM prensa WHERE evento_id=? ORDER BY id LIMIT 30", (evento_id,))
    return {"evento": ev[0], "deliberaciones": de, "alertas": al, "prensa": pr}


def consejos_recientes(limit=8, pais=None):
    w = "AND e.pais=?" if pais else ""
    ids = _rows("SELECT e.id FROM eventos e WHERE EXISTS (SELECT 1 FROM deliberaciones d WHERE d.evento_id=e.id) "
                f"{w} ORDER BY e.creado DESC LIMIT ?", ((pais, limit) if pais else (limit,)))
    return [consejo(r["id"]) for r in ids]


def consejos_hoy(pais):
    return _rows("SELECT COUNT(*) n FROM alertas WHERE pais=? AND modo='real' AND etapa='consejo' AND t >= ?",
                 (pais, (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)).isoformat()))[0]["n"]


def resumen_registro():
    return _rows("SELECT modo, etapa, nivel, COUNT(*) n FROM alertas GROUP BY modo, etapa, nivel")
