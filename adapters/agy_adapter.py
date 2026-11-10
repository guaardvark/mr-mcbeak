"""
Antigravity CLI Adapter.
Runs `agy -p --output-format json`, resuming the same conversation on later turns.
"""
import shutil
from typing import Optional, Dict, Any
from adapters.base_adapter import BaseAdapter


class AntigravityAdapter(BaseAdapter):
    """
    Adapter for Google DeepMind Antigravity CLI (agy).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="Antigravity", target_id="antigravity", config=config)
        self.cli_binary = shutil.which(config.get("cli_cmd") or "agy") or "agy"

    def execute(self, prompt: str, cwd: Optional[str] = None, context=None) -> str:
        if not prompt or not prompt.strip():
            return "Antigravity online. Ready for tasks."

        cmd = [self.cli_binary, "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        if self.readonly:
            cmd += ["--mode", "plan"]
        if self.session_id:
            cmd += ["--conversation", self.session_id]

        raw = self.run_subprocess(cmd, cwd=cwd, context=context)
        text, session = self.parse_json_reply(raw, ("response", "result"), ("conversation_id",))
        if session:
            self.session_id = session
        return text or f"{self.name} did not answer."
