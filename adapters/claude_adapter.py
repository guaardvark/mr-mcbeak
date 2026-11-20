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
from core.approval import rule_prefix, subcommands
from core.council import SPOKEN_SYSTEM

logger = logging.getLogger(__name__)

# With full permissions a voice turn runs in Claude Code's auto mode: ordinary
# work goes ahead and its classifier blocks destructive actions. These are held
# for a spoken yes as well (auto mode allows pushing to the working repository,
# and a misheard sentence should not delete or rewrite anything unasked).
CONFIRM_FIRST = [
    "Bash(git push:*)", "Bash(git reset --hard:*)", "Bash(git clean:*)",
    "Bash(git checkout --:*)", "Bash(git restore:*)", "Bash(git branch -D:*)",
    "Bash(git stash drop:*)", "Bash(git rm:*)", "Bash(rm:*)", "Bash(sudo:*)",
]


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
        self.confirm_first = list(config.get("confirm_first") or CONFIRM_FIRST)
        # Actions refused on the last turn, as [{"tool", "detail"}], and actions
        # the user approved by voice for the next one
        self.last_held = []
        self.approve_next = []

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

    @staticmethod
    def _held(raw: str):
        try:
            denials = json.loads(raw).get("permission_denials") or []
        except (ValueError, AttributeError):
            return []
        held = []
        for d in denials:
            tool_input = d.get("tool_input") or {}
            detail = tool_input.get("command") or tool_input.get("file_path") or tool_input.get("notebook_path") or ""
            item = {"tool": d.get("tool_name") or "tool", "detail": str(detail)}
            if item not in held:
                held.append(item)
        return held

    @staticmethod
    def _approved(rule: str, approved) -> bool:
        """An ask rule is lifted for the approved run when it covers an approved command."""
        prefix = rule_prefix(rule)
        return bool(prefix) and any(
            sub.startswith(prefix) for a in approved if a["tool"] == "Bash" for sub in subcommands(a["detail"]))

    @staticmethod
    def _allow_rules(approved):
        rules = []
        for a in approved:
            if a["tool"] == "Bash":
                rules.append(f"Bash({a['detail']})")
            elif a["detail"]:
                rules.append(f"{a['tool']}({a['detail']})")
            else:
                rules.append(a["tool"])
        return rules

    def reset_session(self):
        super().reset_session()
        self.last_held, self.approve_next = [], []

    # Replies can be spoken while they are being written (execute(on_text=...))
    streams = True
    # The spoken-reply hint rides as --append-system-prompt, not in each question
    spoken_system = True

    @staticmethod
    def _stream_handler(on_text):
        """Turns stream-json lines into on_text(piece) calls for the main
        conversation's text; on_text(None) once a tool was refused for lack of
        permission, since what follows explains the refusal and the cockpit asks
        its own question instead."""
        def handle(line):
            try:
                event = json.loads(line)
            except ValueError:
                return
            if event.get("parent_tool_use_id"):
                return  # a subagent's own text
            if event.get("type") == "stream_event":
                inner = event.get("event") or {}
                delta = inner.get("delta") or {}
                if inner.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                    on_text(delta.get("text") or "")
            elif event.get("type") == "user":
                for block in ((event.get("message") or {}).get("content") or []):
                    if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                        body = json.dumps(block.get("content", ""))
                        if "permission" in body.lower() or "haven't granted" in body.lower():
                            on_text(None)
        return handle

    @staticmethod
    def _result_line(raw: str) -> str:
        """The final {"type": "result", ...} event of a stream-json run."""
        for line in reversed((raw or "").splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    if json.loads(line).get("type") == "result":
                        return line
                except ValueError:
                    continue
        return raw

    def execute(self, prompt: str, cwd: Optional[str] = None, context=None, on_text=None) -> str:
        self.last_held = []
        if not prompt or not prompt.strip():
            return "Yes, how can I help you?"

        # The prompt goes first: --mcp-config takes several values and would
        # swallow a prompt placed after it
        cmd = [self.cli_binary, "-p", prompt]
        cmd += (["--output-format", "stream-json", "--verbose", "--include-partial-messages"]
                if on_text else ["--output-format", "json"])
        if self.model:
            cmd += ["--model", self.model]
        if self.effort:
            cmd += ["--effort", str(self.effort)]
        approved, self.approve_next = self.approve_next, []
        if self.readonly:
            cmd += ["--permission-mode", "plan"]
        else:
            # Before 2026-11-10 "full" passed no mode, and `claude -p` without one
            # refuses every edit and command (Manual mode), so voice turns were
            # read-only in practice
            cmd += ["--permission-mode", "auto"]
            asks = [r for r in self.confirm_first if not self._approved(r, approved)]
            if asks:
                cmd += ["--settings", json.dumps({"permissions": {"ask": asks}})]
            if approved:
                cmd += ["--allowedTools", *self._allow_rules(approved)]
        if self.session_id:
            cmd += ["--resume", self.session_id]
        cmd += ["--append-system-prompt",
                SPOKEN_SYSTEM.format(name=self.name) + (f" Your role on this team: {self.role}" if self.role else "")]
        cmd += self._mcp_args()

        if on_text:
            raw = self._result_line(self.run_streaming(cmd, self._stream_handler(on_text), cwd=cwd, context=context))
        else:
            raw = self.run_subprocess(cmd, cwd=cwd, context=context)
        self.last_held = self._held(raw)
        text, session = self.parse_json_reply(raw, ("result",), ("session_id",))
        if session:
            self.session_id = session
        return text or f"{self.name} did not answer."
