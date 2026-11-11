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

logger = logging.getLogger(__name__)


class AdapterDispatcher:
    """
    Registry and router for all assistant adapters.
    """

    def __init__(self, config_assistants: Dict[str, dict]):
        self._locks = {}
        self.adapters: Dict[str, BaseAdapter] = {}

        # Map target IDs to adapter classes
        adapter_classes = {
            "claude": ClaudeAdapter,
            "grok": GrokAdapter,
            "antigravity": AntigravityAdapter,
            "agy": AntigravityAdapter,
            "cursor": CursorAdapter,
            "guaardvark": GuaardvarkAdapter,
            "codex": CodexAdapter,
        }

        for key, cfg in config_assistants.items():
            target_id = cfg.get("target_id", key)
            cls = adapter_classes.get(target_id) or adapter_classes.get(key)
            if cls:
                adapter = cls(cfg)
                self._locks[adapter] = threading.Lock()
                self.adapters[target_id] = adapter
                self.adapters[key] = adapter
                logger.info(f"Registered assistant adapter: {adapter.name} ({target_id})")

    def reset_sessions(self, target):
        adapters = (set(self.adapters.values()) if target == "team" else {self.adapters[target]})
        for adapter in adapters:
            with self._locks[adapter]:
                adapter.reset_session()

    def dispatch(self, target_id: str, prompt: str, cwd: Optional[str] = None, context=None,
                 on_text=None) -> Tuple[str, str, int]:
        """
        Execute prompt against target adapter.
        Returns: (response_text, voice_file, speaker_id)
        """
        target = target_id.lower().strip()
        adapter = self.adapters.get(target)

        if not adapter:
            # Fallback to claude or first available
            adapter = self.adapters.get("claude") or next(iter(self.adapters.values()), None)

        if not adapter:
            return f"No assistant adapter found for target: {target_id}", "en_US-ryan-high.onnx", 0

        logger.info(f"Dispatching prompt to [{adapter.name}]: '{prompt[:60]}...'")
        if context and context.cancelled.is_set():
            return "", adapter.voice_file, adapter.voice_speaker
        with self._locks[adapter]:
            if context and context.cancelled.is_set():
                return "", adapter.voice_file, adapter.voice_speaker
            if on_text and getattr(adapter, "streams", False):
                response = adapter.execute(prompt, cwd=cwd, context=context, on_text=on_text)
            else:
                response = adapter.execute(prompt, cwd=cwd, context=context)
        if context:
            context.timing("backend_response")
            if context.cancelled.is_set():
                response = ""

        return response, adapter.voice_file, adapter.voice_speaker
