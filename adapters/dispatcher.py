"""
Master Dispatcher for AI Adapters.
Routes prompts to the designated assistant and returns output + voice configuration.
"""
import logging
import threading
from typing import Dict, Any, Optional, Tuple

from adapters.base_adapter import BaseAdapter
from adapters.claude_adapter import ClaudeAdapter
from adapters.grok_adapter import GrokAdapter
from adapters.agy_adapter import AntigravityAdapter
from adapters.cursor_adapter import CursorAdapter
from adapters.guaardvark_adapter import GuaardvarkAdapter
from adapters.codex_adapter import CodexAdapter
from adapters.ollama_adapter import OllamaAdapter

logger = logging.getLogger(__name__)


ADAPTER_CLASSES = {
    "claude": ClaudeAdapter,
    "grok": GrokAdapter,
    "antigravity": AntigravityAdapter,
    "agy": AntigravityAdapter,
    "cursor": CursorAdapter,
    "guaardvark": GuaardvarkAdapter,
    "codex": CodexAdapter,
    "ollama": OllamaAdapter,
}


class AdapterDispatcher:
    """
    Registry and router for the team's adapters, one per member. A member's
    kind (`kind`, or the older `target_id`) picks the adapter class, so a team
    can hold several members of one kind, each with its own conversation.
    """

    def __init__(self, config_assistants: Dict[str, dict], sessions=None):
        self._locks = {}
        self.adapters: Dict[str, BaseAdapter] = {}
        # Optional core.sessions.SessionStore: conversations carry on after a restart
        self.sessions = sessions
        # "group:<name>" -> member keys, for "new conversation" said to a group
        self.groups: Dict[str, list] = {}
        self._restored = set()
        self.update(config_assistants)

    def update(self, config_assistants: Dict[str, dict]):
        """Make the registry match the members given. A member whose settings
        did not change keeps its adapter, and with it its conversation.
        Returns (added keys, removed keys)."""
        current = {key: a for key, a in self.adapters.items() if getattr(a, "member_key", None) == key}
        adapters: Dict[str, BaseAdapter] = {}
        added = []
        for key, cfg in config_assistants.items():
            kind = cfg.get("kind") or cfg.get("target_id", key)
            cls = ADAPTER_CLASSES.get(kind) or ADAPTER_CLASSES.get(key)
            if not cls:
                logger.warning(f"Team member {key!r} has an unknown kind {kind!r}; left out")
                continue
            adapter = current.get(key)
            if adapter is None or adapter.config != cfg or type(adapter) is not cls:
                adapter = cls(dict(cfg))
                adapter.config = cfg
                adapter.member_key = key
                self._locks[adapter] = threading.Lock()
                if self.sessions and self.sessions.restore(key, adapter):
                    self._restored.add(adapter)
                added.append(key)
                logger.info(f"Registered assistant adapter: {adapter.name} ({kind})")
            adapters[key] = adapter
        # Older configs name a member by its kind's id as well ("agy"); a member's
        # own key always wins
        for key, cfg in config_assistants.items():
            alias = cfg.get("target_id")
            if alias and key in adapters:
                adapters.setdefault(alias, adapters[key])
        removed = [key for key in current if key not in config_assistants]
        self.adapters = adapters
        live = set(adapters.values())
        for adapter in set(current.values()) - live:
            adapter.close()
        self._locks = {a: lock for a, lock in self._locks.items() if a in live}
        return added, removed

    def members(self):
        """Each adapter once, in team order."""
        return list(dict.fromkeys(self.adapters.values()))

    def reset_sessions(self, target):
        if target == "team":
            adapters = set(self.adapters.values())
        elif target in self.groups:
            adapters = {self.adapters[m] for m in self.groups[target] if m in self.adapters}
        else:
            adapters = {self.adapters[target]}
        for adapter in adapters:
            with self._locks[adapter]:
                adapter.reset_session()
                self._restored.discard(adapter)
                if self.sessions:
                    self.sessions.forget(adapter.member_key)

    def _run(self, adapter, prompt, cwd, context, on_text):
        if on_text and getattr(adapter, "streams", False):
            return adapter.execute(prompt, cwd=cwd, context=context, on_text=on_text)
        return adapter.execute(prompt, cwd=cwd, context=context)

    def dispatch(self, target_id: str, prompt: str, cwd: Optional[str] = None, context=None,
                 on_text=None) -> Tuple[str, str, int]:
        """
        Execute prompt against target adapter.
        Returns: (response_text, voice_file, speaker_id)
        """
        target = target_id.lower().strip()
        adapter = self.adapters.get(target)

        if not adapter:
            return f"There is no team member called {target_id}.", "en_US-libritts-high.onnx", 0

        logger.info(f"Dispatching prompt to [{adapter.name}]: '{prompt[:60]}...'")
        if context and context.cancelled.is_set():
            return "", adapter.voice_file, adapter.voice_speaker
        with self._locks[adapter]:
            if context and context.cancelled.is_set():
                return "", adapter.voice_file, adapter.voice_speaker
            restored = adapter in self._restored
            self._restored.discard(adapter)
            response = self._run(adapter, prompt, cwd, context, on_text)
            if restored and adapter.last_error and not (context and context.cancelled.is_set()):
                # The conversation remembered from the last run could not be
                # picked up (deleted, or from another folder): start a new one
                logger.info(f"[{adapter.name}] Saved conversation not resumed ({adapter.last_error[:80]}); starting fresh")
                adapter.reset_session()
                response = self._run(adapter, prompt, cwd, context, on_text)
            if self.sessions and not adapter.last_error:
                self.sessions.save(adapter.member_key, adapter)
        if context:
            context.timing("backend_response")
            if context.cancelled.is_set():
                response = ""

        return response, adapter.voice_file, adapter.voice_speaker
