"""
Grok CLI Adapter.
Runs `grok -p --output-format json` without the subagent swarm, resuming the
same session on later turns.
"""
import shutil
from typing import Optional, Dict, Any
from adapters.base_adapter import BaseAdapter


class GrokAdapter(BaseAdapter):
    """
    Adapter for Grok Build CLI.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="Grok", target_id="grok", config=config)
        self.cli_binary = shutil.which(config.get("cli_cmd") or "grok") or "grok"

    def execute(self, prompt: str, cwd: Optional[str] = None, context=None) -> str:
        if not prompt or not prompt.strip():
            return "Grok here, what's on your mind?"

        cmd = [
            self.cli_binary,
            "--no-subagents",
            "--max-turns", "6",
            "--output-format", "json",
        ]
        if self.model:
            cmd += ["--model", self.model]
        if self.readonly:
            # Grok has no read-only mode flag; deny the tools that change things
            cmd += ["--deny", "Write", "--deny", "Edit", "--deny", "Bash"]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        cmd += ["-p", prompt]

        raw = self.run_subprocess(cmd, cwd=cwd, context=context)
        return self._final_text(raw)

    def _final_text(self, raw: str) -> str:
        text, session = self.parse_json_reply(
            raw, ("text", "result", "response", "message", "content"), ("sessionId", "session_id")
        )
        if session:
            self.session_id = session
        return text or f"{self.name} did not answer."
