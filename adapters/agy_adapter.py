"""
Antigravity CLI Adapter.
Invokes Antigravity CLI: `agy -p "<prompt>"`
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
        self.cli_binary = shutil.which("agy") or "~/.local/bin/agy"

    def execute(self, prompt: str, cwd: Optional[str] = None) -> str:
        if not prompt or not prompt.strip():
            return "Antigravity online. Ready for tasks."

        cmd = [self.cli_binary, "-p", prompt]
        return self.run_subprocess(cmd, cwd=cwd, timeout=120)
