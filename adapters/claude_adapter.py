"""
Claude Code CLI Adapter.
Invokes Claude Code in print/headless mode: `claude -p "<prompt>"`
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
        self.cli_binary = shutil.which("claude") or "~/.local/bin/claude"

    def execute(self, prompt: str, cwd: Optional[str] = None) -> str:
        if not prompt or not prompt.strip():
            return "Yes, how can I help you?"

        cmd = [self.cli_binary, "-p", prompt]
        return self.run_subprocess(cmd, cwd=cwd, timeout=120)
