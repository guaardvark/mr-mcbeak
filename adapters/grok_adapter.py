"""
Grok CLI Adapter.
Invokes Grok in print/headless mode without spawning the agent team.
"""
import json
import shutil
from typing import Optional, Dict, Any
from adapters.base_adapter import BaseAdapter


class GrokAdapter(BaseAdapter):
    """
    Adapter for Grok Build CLI.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="Grok", target_id="grok", config=config)
        self.cli_binary = shutil.which("grok") or "~/.grok/bin/grok"

    def execute(self, prompt: str, cwd: Optional[str] = None) -> str:
        if not prompt or not prompt.strip():
            return "Grok here, what's on your mind?"

        # Voice needs the final reply only — no subagent swarm, no live traces
        cmd = [
            self.cli_binary,
            "--no-subagents",
            "--max-turns", "6",
            "--output-format", "json",
            "-p",
            prompt,
        ]
        raw = self.run_subprocess(cmd, cwd=cwd, timeout=120)
        return self._final_text(raw)

    @staticmethod
    def _final_text(raw: str) -> str:
        blob = (raw or "").strip()
        if not blob:
            return blob
        start = blob.find("{")
        if start == -1:
            return blob
        try:
            data = json.loads(blob[start:])
        except json.JSONDecodeError:
            return blob
        if not isinstance(data, dict):
            return blob
        for key in ("result", "response", "text", "message", "content"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return blob
