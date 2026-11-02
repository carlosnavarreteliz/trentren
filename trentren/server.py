"""Servidor de Trentren: web + API + stream en vivo del consejo de agentes.

  python3 -m trentren            # vigía (Chile, Japón, EE.UU.) + web en http://127.0.0.1:8787
"""
import json
import mimetypes
import os
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import db
from .config import HOST, PORT, WEB
from .council import Consejo, ESCENARIOS, verificar_meta
from .regions import PAISES

C = None
SIM_POR_HORA = int(os.environ.get("TRENTREN_SIM_POR_HORA", "6"))
_sims = deque()
_sim_lock = threading.Lock()


def _permitir_sim():
    """Simulacros públicos: uno a la vez y un máximo por hora (cada uno cuesta llamadas a Meta)."""
    with _sim_lock:
        t = time.time()
        while _sims and t - _sims[0] > 3600:
            _sims.popleft()
        if C.vivo or C.tasks.qsize():
            return "El consejo está deliberando; espera a que termine."
        if len(_sims) >= SIM_POR_HORA:
            return f"Límite de {SIM_POR_HORA} simulaciones por hora alcanzado; mira los consejos ya grabados."
        _sims.append(t)
        return None


class H(BaseHTTPRequestHandler):
    server_version = "Trentren/2.0"

    def log_message(self, fmt, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(min(n, 10000)) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        pais = q.get("pais", "chile")
        if pais not in PAISES:
            pais = "chile"
        if u.path == "/api/estado":
            return self._json(C.publico(pais))
        if u.path == "/api/momento":
            return self._json(C.momento(pais, q.get("t", "")))
        if u.path == "/api/consejo":
            return self._json(db.consejo(q.get("id", "")) or {"error": "no existe"})
        if u.path == "/api/meta":
            return self._json(verificar_meta())
        if u.path == "/api/stream":
            return self._stream()
        f = (WEB / ("index.html" if u.path in ("/", "") else u.path.lstrip("/"))).resolve()
        if WEB.resolve() not in f.parents or not f.is_file():
            self.send_error(404)
            return
        b = f.read_bytes()
        self.send_response(200)
        ct = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        self.send_header("Content-Type", ct + ("; charset=utf-8" if f.suffix in (".html", ".js", ".css", ".json") else ""))
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(b)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        sq = C.bus.sub()
        try:
            while True:
                try:
                    self.wfile.write(f"data: {json.dumps(sq.get(timeout=20), ensure_ascii=False, default=str)}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": latido\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            C.bus.unsub(sq)

    def do_POST(self):
        u = urlparse(self.path)
        b = self._body()
        pais = b.get("pais", "chile")
        if pais not in PAISES:
            return self._json({"error": "país desconocido"}, 400)
        if u.path == "/api/simulacro":
            if b.get("escenario") not in ESCENARIOS[pais]:
                return self._json({"error": "escenario desconocido"}, 400)
            if (err := _permitir_sim()):
                return self._json({"error": err}, 429)
            return self._json({"ok": True, "evento": C.simulacro(pais, b["escenario"])})
        if u.path == "/api/revivir":
            if not any(g["t"] == b.get("t") for g in C.hist[pais]["grandes"]):
                return self._json({"error": "sismo histórico desconocido"}, 400)
            if (err := _permitir_sim()):
                return self._json({"error": err}, 429)
            return self._json({"ok": True, "evento": C.revivir(pais, b["t"])})
        self.send_error(404)


def main():
    global C
    C = Consejo()
    C.start()
    srv = ThreadingHTTPServer((HOST, PORT), H)
    srv.daemon_threads = True
    print(f"Trentren vigilando Chile, Japón y EE.UU. · http://{HOST}:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
