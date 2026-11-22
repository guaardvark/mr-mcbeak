"""Guaardvark's API for Mr. McBeak, under /api/mcbeak.

    GET       /status     the listener's /health, or {"running": false}
    GET, PUT  /team       ~/.config/mcbeak/team.yaml, through core/team.py's rules
    GET, PUT  /settings   ~/.config/mcbeak/settings.yaml, checked as a start checks it

The files are read and written by ../bridge.py in a process of its own, run
with Mr. McBeak's interpreter. Starting and stopping the listener is the
plugin's job (/api/plugins/mcbeak/start and /stop).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from flask import Blueprint, request

from backend.utils import auth_guard
from backend.utils.response_utils import error_response, success_response

mcbeak_bp = Blueprint("mcbeak", __name__, url_prefix="/api/mcbeak")

# The extension folder is usually a symlink into the checkout; resolve it to find the app.
MCBEAK_ROOT = Path(__file__).resolve().parents[2]
BRIDGE = MCBEAK_ROOT / "guaardvark" / "bridge.py"
STATUS_URL = f"http://127.0.0.1:{int(os.environ.get('MCBEAK_STATUS_PORT') or 8214)}/health"
# The team and settings decide which programs a spoken word may run and with
# what permissions, so they change only from a browser on this machine.
NOT_HERE = "The voice team is changed at the machine itself. Open this page in a browser on that machine."
# No proxy from the environment: the status endpoint is on this machine.
_local = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _python() -> str:
    own = MCBEAK_ROOT / ".venv" / "bin" / "python"
    return str(own) if own.is_file() else sys.executable


def _bridge(command: str, payload: dict | None = None):
    try:
        done = subprocess.run(
            [_python(), str(BRIDGE), command],
            input=json.dumps(payload) if payload is not None else None,
            capture_output=True, text=True, timeout=20, cwd=str(MCBEAK_ROOT),
        )
        result = json.loads(done.stdout)
    except subprocess.TimeoutExpired:
        return error_response("Mr. McBeak's settings helper did not answer in time.", 504)
    except (OSError, ValueError) as err:
        return error_response(f"Mr. McBeak's settings helper failed: {err}", 500)
    if not result.get("ok"):
        return error_response(result.get("error") or "The change was refused.", 422)
    return success_response(result["data"])


@mcbeak_bp.route("/status", methods=["GET"])
def status():
    try:
        with _local.open(STATUS_URL, timeout=2) as reply:
            health = json.loads(reply.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return success_response({"running": False})
    if not isinstance(health, dict):
        return success_response({"running": False})
    return success_response({**health, "running": True})


@mcbeak_bp.route("/team", methods=["GET"])
def get_team():
    return _bridge("team-get")


@mcbeak_bp.route("/team", methods=["PUT"])
def put_team():
    if not auth_guard.request_is_from_this_machine():
        return error_response(NOT_HERE, 403)
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("members"), list):
        return error_response('The body needs a "members" list.', 400)
    return _bridge("team-put", {"members": body["members"], "groups": body.get("groups")})


@mcbeak_bp.route("/settings", methods=["GET"])
def get_settings():
    return _bridge("settings-get")


@mcbeak_bp.route("/settings", methods=["PUT"])
def put_settings():
    if not auth_guard.request_is_from_this_machine():
        return error_response(NOT_HERE, 403)
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not ("text" in body or "settings" in body):
        return error_response('The body needs the settings as "text" or as a "settings" object.', 400)
    return _bridge("settings-put", {k: body[k] for k in ("text", "settings") if k in body})
