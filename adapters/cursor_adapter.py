"""
Cursor Agent CLI Adapter.
Runs `agent -p --output-format json`, resuming the same chat on later turns.
"""
import shutil
from typing import Optional, Dict, Any
from adapters.base_adapter import BaseAdapter


class CursorAdapter(BaseAdapter):
    """
    Adapter for Cursor Agent CLI.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="Cursor Agent", target_id="cursor", config=config)
        self.cli_binary = (
            shutil.which(config.get("cli_cmd") or "agent") or shutil.which("cursor-agent") or "agent"
        )

    def execute(self, prompt: str, cwd: Optional[str] = None, context=None) -> str:
        if not prompt or not prompt.strip():
            return "Cursor Agent ready. What code should we inspect?"

        # "auto" is the only model a free Cursor plan accepts in print mode
        cmd = [self.cli_binary, "-p", "--output-format", "json", "--model", self.model or "auto"]
        if self.readonly:
            cmd += ["--mode", "ask"]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        cmd.append(prompt)

        raw = self.run_subprocess(cmd, cwd=cwd, context=context)
        text, session = self.parse_json_reply(raw, ("result",), ("session_id",))
        if session:
            self.session_id = session
        return text or f"{self.name} did not answer."
