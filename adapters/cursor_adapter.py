"""
Cursor Agent CLI Adapter.
Invokes Cursor Agent: `agent -p "<prompt>"` or `cursor-agent -p "<prompt>"`
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
        self.cli_binary = shutil.which("agent") or shutil.which("cursor-agent") or "~/.local/bin/agent"

    def execute(self, prompt: str, cwd: Optional[str] = None) -> str:
        if not prompt or not prompt.strip():
            return "Cursor Agent ready. What code should we inspect?"

        cmd = [self.cli_binary, "--trust", "-p", prompt]
        return self.run_subprocess(cmd, cwd=cwd, timeout=120)
