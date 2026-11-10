"""
Claude Code CLI Adapter.
Runs `claude -p --output-format json`, resumes the same session on later turns.
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class ClaudeAdapter(BaseAdapter):
    """
    Adapter for Anthropic's Claude Code CLI.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="Claude Code", target_id="claude", config=config)
        self.cli_binary = shutil.which(config.get("cli_cmd") or "claude") or "claude"
        # low | medium | high | xhigh | max; unset keeps Claude Code's own default
        self.effort = config.get("effort")
        # MCP servers for voice turns: "all" (Claude Code's normal set), "none", or
        # a list of user-scope server names. Measured 2026-11-10: the full set
        # (cloud connectors plus a local Guaardvark server) added ~1.5 s and ~$0.05
        # to every turn; with "none" a short Opus turn took 2.2 s instead of 3.6 s.
        self.mcp = config.get("mcp", "all")

    def _mcp_args(self):
        if self.mcp in (None, "all"):
            return []
        if self.mcp == "none":
            return ["--strict-mcp-config"]
        wanted = [self.mcp] if isinstance(self.mcp, str) else list(self.mcp)
        try:
            known = json.loads((Path.home() / ".claude.json").read_text()).get("mcpServers", {})
        except (OSError, ValueError):
            known = {}
        chosen = {name: known[name] for name in wanted if name in known}
        missing = [name for name in wanted if name not in known]
        if missing:
            logger.warning(f"[{self.name}] MCP servers not found in user config, skipped: {missing}")
        args = ["--strict-mcp-config"]
        if chosen:
            args += ["--mcp-config", json.dumps({"mcpServers": chosen})]
        return args

    def execute(self, prompt: str, cwd: Optional[str] = None, context=None) -> str:
        if not prompt or not prompt.strip():
            return "Yes, how can I help you?"

        # The prompt goes first: --mcp-config takes several values and would
        # swallow a prompt placed after it
        cmd = [self.cli_binary, "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", str(self.effort)]
        if self.readonly:
            cmd += ["--permission-mode", "plan"]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        cmd += self._mcp_args()

        raw = self.run_subprocess(cmd, cwd=cwd, context=context)
        text, session = self.parse_json_reply(raw, ("result",), ("session_id",))
        if session:
            self.session_id = session
        return text or f"{self.name} did not answer."
