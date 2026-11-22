"""A status endpoint for a host program that starts and watches the listener.

Off unless a port is given (`--status-port N` or $MCBEAK_STATUS_PORT). It
binds to 127.0.0.1 only and answers one request, `GET /health`:

    {"status": "ok", "version": "0.1.0", "listening": true,
     "team": ["Claude Code", "Grok"], "integration": {"mode": "standalone"}}

`status` is "starting" until the microphone is open. Only the fields above are
ever sent: nothing that was heard or said goes through here.
"""
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Mapping, Optional

from core.version import __version__

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT_ENV = "MCBEAK_STATUS_PORT"


def integration_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """"guaardvark" when a Guaardvark install started this copy, else "standalone"."""
    env = os.environ if env is None else env
    return "guaardvark" if env.get("GUAARDVARK_ROOT") else "standalone"


def port_from(flag: Optional[int] = None, env: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """The port asked for on the command line or in the environment, or None for no endpoint."""
    env = os.environ if env is None else env
    raw = flag if flag is not None else (env.get(PORT_ENV) or "").strip()
    if raw in (None, ""):
        return None
    try:
        port = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"The status port must be a number, not {raw!r}.")
    if not 0 <= port <= 65535:
        raise ValueError(f"The status port must be between 0 and 65535, not {port}.")
    return port


def health(state: Mapping) -> dict:
    """The /health document. `state` supplies status, listening and team; any
    other key it carries is left out."""
    return {
        "status": str(state.get("status") or "starting"),
        "version": __version__,
        "listening": bool(state.get("listening")),
        "team": [str(name) for name in (state.get("team") or [])],
        "integration": {"mode": integration_mode()},
    }


class StatusServer:
    """Serves /health on the loopback interface from a daemon thread."""

    def __init__(self, port: int, state: Callable[[], Mapping]):
        self._state = state
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.split("?", 1)[0] != "/health":
                    return self._send(404, {"error": "not found"})
                try:
                    body = health(owner._state())
                except Exception:
                    logger.exception("Status endpoint could not read the listener's state")
                    return self._send(500, {"error": "status unavailable"})
                self._send(200, body)

            def _send(self, code: int, body: dict):
                data = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, fmt, *args):
                logger.debug("status endpoint: " + fmt, *args)

        self._server = ThreadingHTTPServer((HOST, port), Handler)
        self._server.daemon_threads = True
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def start(self) -> int:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="status-server")
        self._thread.start()
        return self.port

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
