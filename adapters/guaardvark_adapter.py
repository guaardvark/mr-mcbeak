"""
Guaardvark System Adapter.
Invokes local Guaardvark via REST API or CLI (`guaardvark ask "<prompt>"`).
"""
import json
import uuid
import shutil
import logging
from typing import Optional, Dict, Any
from adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class GuaardvarkAdapter(BaseAdapter):
    """
    Adapter for local Guaardvark autonomous AI operating system.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="Guaardvark", target_id="guaardvark", config=config)
        self.cli_binary = shutil.which(config.get("cli_cmd") or "guaardvark") or "guaardvark"
        self.api_url = config.get("api_url", "http://127.0.0.1:5000/api/enhanced-chat")
        self.session_id = uuid.uuid4().hex
        self.use_rag = bool(config.get("use_rag", False))  # spoken turns want speed, not a document search

    def reset_session(self):
        self.session_id = uuid.uuid4().hex

    def unavailable(self):
        return None  # the API or the CLI may come up later; a failed turn says so

    def execute(self, prompt: str, cwd: Optional[str] = None, context=None) -> str:
        if not prompt or not prompt.strip():
            return "Guaardvark online. Standing by for instructions."

        if self.readonly:
            return "Guaardvark is unavailable in read-only mode: its API and CLI do not enforce this restriction."
        if context and context.cancelled.is_set():
            return ""

        # Try REST API first (faster if backend is running)
        self.last_error = None
        try:
            resp = self.post_json(
                self.api_url,
                {"message": prompt, "session_id": self.session_id or "voice_cockpit", "use_rag": self.use_rag},
                context=context,
            )
            if resp is None or (context and context.cancelled.is_set()):
                return ""
            if resp.status_code == 200:
                data = resp.json()
                inner = data.get("data") if isinstance(data.get("data"), dict) else data
                reply = inner.get("response") or inner.get("message") or inner.get("text")
                if isinstance(reply, str) and reply.strip():
                    return reply.strip()
                logger.warning(f"Guaardvark API gave no reply text: {str(data)[:200]}")
            else:
                logger.warning(f"Guaardvark API HTTP {resp.status_code}, falling back to CLI")
        except Exception as e:
            logger.debug(f"Guaardvark API offline ({e}), falling back to CLI")

        if context and context.cancelled.is_set():
            return ""
        # Fallback to CLI
        cmd = [self.cli_binary, "--json", "--timeout", "30", "ask", prompt]
        raw = self.run_subprocess(cmd, cwd=cwd, timeout=60, context=context)
        return self._cli_reply(raw)

    @staticmethod
    def _cli_reply(raw: str) -> str:
        """`guaardvark --json ask` prints {"status": ..., "data": {"response": ..., "session_id": ...}}."""
        blob = (raw or "").strip()
        start = blob.find("{")
        if start == -1:
            return blob
        try:
            data = json.loads(blob[start:])
        except json.JSONDecodeError:
            return blob
        inner = data.get("data") if isinstance(data.get("data"), dict) else data
        for key in ("response", "message", "text"):
            val = inner.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return blob
