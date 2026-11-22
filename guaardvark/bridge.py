#!/usr/bin/env python3
"""Reads and writes the team and settings files for the Guaardvark API in api/.

That API runs this as a separate process, so none of this app's modules are
loaded into the Guaardvark server:

    python guaardvark/bridge.py team-get | team-put | settings-get | settings-put

A put reads one JSON object on stdin. One JSON object is printed:
{"ok": true, "data": ...} or {"ok": false, "error": "what to fix"}.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from core import local_settings, team  # noqa: E402
from core.diagnostics import validate_config  # noqa: E402


def shipped_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}


def team_view(config: dict) -> dict:
    members = team.current_members(config)
    resolved = team.resolve([dict(m) for m in members], config, ROOT)
    rows = []
    for member, entry in zip(members, resolved.values()):
        rows.append({
            "member": member,   # what the team file holds; send these back to change the team
            "name": entry.get("name", member.get("name")),
            "kind": entry.get("kind"),
            "voice": entry.get("voice"),
            "voice_speaker": entry.get("voice_speaker", 0),
            "permissions": entry.get("permissions"),
            "model": entry.get("model") or "",
            "folder": entry.get("folder") or entry.get("workspace") or "",
            "missing_program": team.missing_program(entry),
        })
    return {
        "path": str(team.team_path()),
        "exists": team.read_team() is not None,
        "kinds": list(team.KINDS),
        "cloud_kinds": list(team.CLOUD_KINDS),
        "members": rows,
        "groups": team.read_groups(),
    }


def team_put(payload: dict) -> dict:
    config = local_settings.apply(shipped_config())
    team.replace_team(config, payload.get("members"), payload.get("groups"), root=ROOT)
    return team_view(config)


def settings_view() -> dict:
    path = local_settings.settings_path()
    text = local_settings.read_text(path)
    view = {"path": str(path), "exists": path.exists(), "text": text, "settings": None, "problem": None}
    try:
        view["settings"] = local_settings.parse(text, str(path))
    except ValueError as err:
        view["problem"] = str(err)
    return view


def settings_put(payload: dict) -> dict:
    text = payload.get("text")
    if text is None:
        if not isinstance(payload.get("settings"), dict):
            raise ValueError('Send the settings as "text" (YAML) or as a "settings" object.')
        text = yaml.safe_dump(payload["settings"], sort_keys=False, allow_unicode=True) if payload["settings"] else ""
    if not isinstance(text, str):
        raise ValueError("The settings text must be a string.")
    # The checks a start makes, so a file that would stop the listener from
    # starting is never written. Run on the assistants as config.yaml lists
    # them as well: a team file replaces that list, and a bad value in it
    # would otherwise wait for the day the team file is gone.
    merged = local_settings.merge(shipped_config(), local_settings.parse(text))
    validate_config({**merged, "council": {}})
    validate_config(team.apply(merged, root=ROOT))
    local_settings.write_text(text)
    return settings_view()


COMMANDS = {
    "team-get": lambda payload: team_view(local_settings.apply(shipped_config())),
    "team-put": team_put,
    "settings-get": lambda payload: settings_view(),
    "settings-put": settings_put,
}


def main(argv) -> int:
    if len(argv) != 2 or argv[1] not in COMMANDS:
        print(json.dumps({"ok": False, "error": f"usage: bridge.py {' | '.join(COMMANDS)}"}))
        return 2
    try:
        payload = {}
        if argv[1].endswith("-put"):
            payload = json.load(sys.stdin)
            if not isinstance(payload, dict):
                raise ValueError("The request must be a JSON object.")
        result = {"ok": True, "data": COMMANDS[argv[1]](payload)}
    except (ValueError, OSError) as err:   # TeamError and bad JSON are ValueErrors
        result = {"ok": False, "error": str(err)}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
