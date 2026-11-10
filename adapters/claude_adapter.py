"""
Claude Code CLI Adapter.
Runs `claude -p --output-format json`, resumes the same session on later turns.
"""
import shutil
from typing import Optional, Dict, Any
from adapters.base_adapter import BaseAdapter


class ClaudeAdapter(BaseAdapter):
    """
    Adapter for Anthropic's Claude Code CLI.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="Claude Code", target_id="claude", config=config)
        self.cli_binary = shutil.which(config.get("cli_cmd") or "claude") or "claude"

    def execute(self, prompt: str, cwd: Optional[str] = None, context=None) -> str:
        if not prompt or not prompt.strip():
            return "Yes, how can I help you?"

        cmd = [self.cli_binary, "-p", "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        if self.readonly:
            cmd += ["--permission-mode", "plan"]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        cmd.append(prompt)

        raw = self.run_subprocess(cmd, cwd=cwd, context=context)
        text, session = self.parse_json_reply(raw, ("result",), ("session_id",))
        if session:
            self.session_id = session
        return text or f"{self.name} did not answer."
