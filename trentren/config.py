"""Configuración de Trentren: rutas, claves (desde .env) y constantes."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = Path(__file__).resolve().parent
WEB = PKG / "web"
STATE = PKG / "state"
STATE.mkdir(exist_ok=True)


def _load_env():
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

META_API_KEY = os.environ.get("META_API_KEY", "")
META_MODEL = os.environ.get("META_MODEL", "muse-wonder-1.0")
META_BASE_URL = os.environ.get("META_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
HOST = os.environ.get("TRENTREN_HOST", "127.0.0.1")
PORT = int(os.environ.get("TRENTREN_PORT", "8787"))

# Ritmo del vigía
POLL_SECONDS = 120          # lectura de CSN + USGS
HEARTBEAT_MINUTES = 60      # ronda de vigilancia sin eventos
EVENT_MW = 4.7              # disparador del modelo v4 (Mw-equivalente)
EVENT_CSN = 5.0             # disparador por magnitud CSN
MAX_TOOL_ROUNDS = 10        # tope de pasos por episodio

UA = "Trentren-Vigia/1.0 (+investigacion sismica Chile)"
