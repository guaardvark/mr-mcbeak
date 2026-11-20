"""
Ollama adapter: a team member that is a model served by Ollama, by default on
this machine (http://127.0.0.1:11434), so nothing it hears leaves it. Set
`host` to reach an Ollama on another of your own machines.

It is a voice in the room, not an agent: no tools, no file access. The
conversation is kept here and sent with each turn; the reply is streamed, so
it can be spoken while it is being written.
"""
import json
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from adapters.base_adapter import BaseAdapter
from core.council import SPOKEN_SYSTEM

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://127.0.0.1:11434"
def installed_models(host: str = DEFAULT_HOST, timeout: float = 3.0) -> List[str]:
    """Names of the models this Ollama has. Raises when it cannot be reached."""
    import requests
    data = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout).json()
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


class OllamaAdapter(BaseAdapter):
    streams = True          # replies can be spoken while they are being written
    spoken_system = True    # carries the spoken-reply hint itself, as a system message

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name=config.get("name") or "Ollama", target_id="ollama", config=config)
        self.host = str(config.get("host") or DEFAULT_HOST).rstrip("/")
        # How many past messages are sent along; small models slow down with long histories
        self.history_messages = int(config.get("history_messages", 20))
        # true | false for models that can think before answering; unset leaves the model's default
        self.think = config.get("think")
        self.messages: List[dict] = []
        # No tools, so there is nothing for read-only to restrict
        self.readonly = True

    def reset_session(self):
        self.messages = []

    def unavailable(self) -> Optional[str]:
        if not self.model:
            return "no model is set for it"
        try:
            models = installed_models(self.host)
        except Exception:
            return f"Ollama is not answering at {self.host}"
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in models:
            return f"Ollama has no model called {self.model}"
        return None

    def execute(self, prompt: str, cwd: Optional[str] = None, context=None, on_text=None) -> str:
        if not prompt or not prompt.strip():
            return "I'm here."
        self.last_error = None
        if not self.model:
            self.last_error = "no model is set"
            return f"{self.name} has no model set."
        system = SPOKEN_SYSTEM.format(name=self.name) + (f" {self.role}" if self.role else "")
        payload = {
            "model": self.model, "stream": True,
            "messages": [{"role": "system", "content": system},
                         *self.messages[-self.history_messages:],
                         {"role": "user", "content": prompt}],
        }
        if self.think is not None:
            payload["think"] = bool(self.think)

        import requests
        lines: "queue.Queue" = queue.Queue()
        holder = {}

        def read():
            try:
                with requests.post(f"{self.host}/api/chat", json=payload, stream=True,
                                   timeout=(5, self.timeout)) as resp:
                    holder["response"] = resp
                    if resp.status_code != 200:
                        lines.put(RuntimeError(f"HTTP {resp.status_code}: {resp.text[:150]}"))
                        return
                    for line in resp.iter_lines():
                        if line:
                            lines.put(line)
                lines.put(None)
            except Exception as err:
                lines.put(err)
        threading.Thread(target=read, name="ollama", daemon=True).start()

        parts = []
        deadline = time.monotonic() + self.timeout
        while True:
            if context and context.cancelled.is_set():
                resp = holder.get("response")
                if resp is not None:
                    resp.close()  # stops the model generating
                return ""
            if time.monotonic() >= deadline:
                self.last_error = f"no answer within {self.timeout} seconds"
                return f"Sorry, {self.name} took too long to respond."
            try:
                item = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                logger.error(f"[{self.name}] Ollama error: {item}")
                self.last_error = str(item)[:200]
                return f"{self.name} could not answer: Ollama is not responding." if not parts else "".join(parts)
            try:
                event = json.loads(item)
            except ValueError:
                continue
            if event.get("error"):
                self.last_error = str(event["error"])[:200]
                return f"{self.name} could not answer: {self.last_error[:100]}"
            piece = (event.get("message") or {}).get("content") or ""
            if piece:
                parts.append(piece)
                if on_text:
                    on_text(piece)
            if event.get("done"):
                break
        reply = "".join(parts).strip()
        if reply:
            self.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
            del self.messages[:-2 * self.history_messages]
        return reply or f"{self.name} did not answer."
