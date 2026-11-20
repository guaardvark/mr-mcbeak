"""
Conversations that carry on after a restart.

Each assistant CLI keeps its own conversation and hands back an id for it. The
ids are written to a small file in the state folder, so stopping and starting
the app does not make every assistant forget what was said. Only ids are
stored, never the words. A saved conversation is picked up when the member is
still the same kind, working in the same folder, and was last used recently;
"new conversation" forgets it.
"""
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def default_path() -> Path:
    state = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return state / "mcbeak" / "sessions.json"


class SessionStore:
    def __init__(self, path: Optional[Path] = None, max_age_hours: float = 72.0):
        self.path = Path(path) if path else default_path()
        self.max_age = max_age_hours * 3600.0
        self._lock = threading.Lock()
        self._data = self._read()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as err:
            logger.warning(f"Could not save conversation ids to {self.path}: {err}")

    @staticmethod
    def _folder(adapter) -> str:
        return os.path.realpath(adapter.workspace or os.getcwd())

    def restore(self, key: str, adapter) -> bool:
        """Give the adapter back its saved conversation. True when it took one."""
        with self._lock:
            entry = self._data.get(key)
        if not entry or entry.get("kind") != adapter.target_id or entry.get("folder") != self._folder(adapter):
            return False
        if time.time() - float(entry.get("saved_at", 0)) > self.max_age:
            return False
        state = entry.get("state") or {}
        if not state:
            return False
        adapter.set_state(state)
        logger.info(f"[{adapter.name}] Carrying on the conversation from the last run")
        return True

    def save(self, key: str, adapter):
        state = adapter.get_state()
        if not state:
            return
        entry = {"kind": adapter.target_id, "folder": self._folder(adapter), "state": state, "saved_at": time.time()}
        with self._lock:
            old = self._data.get(key) or {}
            unchanged = all(old.get(k) == entry[k] for k in ("kind", "folder", "state"))
            self._data[key] = entry
            # The time is refreshed at most once a minute when nothing else changed
            if not unchanged or entry["saved_at"] - float(old.get("saved_at", 0)) > 60:
                self._write()

    def forget(self, key: str):
        with self._lock:
            if self._data.pop(key, None) is not None:
                self._write()
