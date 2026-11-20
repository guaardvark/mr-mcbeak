"""
Claude Code CLI Adapter.

A turn is either a fresh `claude -p` that resumes the member's session, or,
once warm() has been called (the listener does this), a message written to one
`claude -p --input-format stream-json` process that stays up between turns.
Measured 2026-11-19 with Haiku on short questions: 1.14 s per turn with a
fresh process, 0.62 s with the process kept alive.
"""
import atexit
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
from adapters.base_adapter import BaseAdapter
from core.approval import rule_prefix, subcommands
from core.council import SPOKEN_SYSTEM
from core.privacy import child_env

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
        # keep_alive: false goes back to a fresh process per turn (each kept
        # process holds a few hundred MB; see README)
        self.keep_alive = bool(config.get("keep_alive", True))
        self._warm = False
        self._live = None            # {"proc", "lines", "stderr", "cwd"} of the kept process
        self._live_lock = threading.RLock()

    # -- the process kept alive between turns ---------------------------------
    def warm(self):
        """Start the long-lived process now, so the first question does not
        wait for it. Without this call every turn is a fresh process."""
        if not self.keep_alive:
            return
        self._warm = True
        atexit.register(self.close)
        self._start_live_soon()

    def close(self):
        """Stop the kept process (the session itself stays on disk)."""
        with self._live_lock:
            live, self._live = self._live, None
        if live and live["proc"].poll() is None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(live["proc"].pid, sig)
                    live["proc"].wait(timeout=0.5)
                    break
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue

    def _start_live_soon(self):
        threading.Thread(target=self._live_process, name="claude-warm", daemon=True).start()

    def _live_process(self, cwd: Optional[str] = None):
        """The kept process for `cwd`, started if there is none. None when it
        cannot be started."""
        working_dir = cwd or self.workspace or os.getcwd()
        with self._live_lock:
            live = self._live
            if live and live["proc"].poll() is None and live["cwd"] == working_dir:
                return live
            self.close()
            cmd = [self.cli_binary, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
                   "--verbose", "--include-partial-messages", *self._turn_flags([])]
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, cwd=working_dir, start_new_session=True, env=child_env(), bufsize=1)
            except OSError as err:
                logger.error(f"[{self.name}] Could not start: {err}")
                return None
            live = {"proc": proc, "lines": queue.Queue(), "stderr": [], "cwd": working_dir}
            threading.Thread(target=lambda: ([live["lines"].put(l) for l in proc.stdout], live["lines"].put(None)),
                             daemon=True).start()
            threading.Thread(target=lambda: live["stderr"].append(proc.stderr.read()), daemon=True).start()
            logger.info(f"[{self.name}] Process kept ready (cwd={working_dir}"
                        f"{', resuming its conversation' if self.session_id else ''})")
            self._live = live
            return live

    def _execute_live(self, live, prompt: str, context=None, on_text=None) -> Optional[str]:
        """One turn through the kept process. Returns the final result event as
        a JSON line, "" when cancelled, or None when the process could not take
        the turn (the caller then runs it the ordinary way)."""
        self.last_error = None
        lines = live["lines"]
        while True:  # startup events and anything left from an abandoned turn
            try:
                if lines.get_nowait() is None:
                    return self._live_died(live)
            except queue.Empty:
                break
        try:
            live["proc"].stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": prompt}}) + "\n")
            live["proc"].stdin.flush()
        except (OSError, ValueError):
            return self._live_died(live)
        logger.info(f"[{self.name}] Asked through the kept process (cwd={live['cwd']})")
        handle = self._stream_handler(on_text) if on_text else None
        deadline = time.monotonic() + self.timeout
        while True:
            cancelled = context and context.cancelled.is_set()
            if cancelled or time.monotonic() >= deadline:
                # The only sure way to stop a turn; the next one resumes the session
                self.close()
                self._start_live_soon()
                if cancelled:
                    return ""
                logger.error(f"[{self.name}] No answer within {self.timeout}s")
                self.last_error = f"no answer within {self.timeout} seconds"
                return json.dumps({"result": f"Sorry, {self.name} took too long to respond."})
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                return self._live_died(live)
            if handle:
                try:
                    handle(line)
                except Exception as err:
                    logger.error(f"[{self.name}] Stream handler error: {err}")
            if '"result"' in line:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "result":
                    if event.get("is_error"):
                        self.last_error = str(event.get("result") or event.get("subtype") or "error")[:200]
                    return line

    def _live_died(self, live) -> Optional[str]:
        """The kept process ended by itself (signed out, session gone, crash)."""
        time.sleep(0.1)
        err = "".join(live["stderr"]).strip()
        logger.warning(f"[{self.name}] Kept process ended (exit {live['proc'].poll()}): {err[:300]}")
        with self._live_lock:
            if self._live is live:
                self._live = None
        self.last_error = err[:200] or "the claude process ended"
        return None

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
        if self._warm:
            self.close()
            self._start_live_soon()

    def set_state(self, state):
        before = self.session_id
        super().set_state(state)
        if self._warm and self.session_id != before:
            self.close()
            self._start_live_soon()

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

    def _turn_flags(self, approved) -> list:
        """Everything on the command line but the prompt and the output format."""
        flags = []
        if self.model:
            flags += ["--model", self.model]
        if self.effort:
            flags += ["--effort", str(self.effort)]
        if self.readonly:
            flags += ["--permission-mode", "plan"]
        else:
            # Before 2026-11-10 "full" passed no mode, and `claude -p` without one
            # refuses every edit and command (Manual mode), so voice turns were
            # read-only in practice
            flags += ["--permission-mode", "auto"]
            asks = [r for r in self.confirm_first if not self._approved(r, approved)]
            if asks:
                flags += ["--settings", json.dumps({"permissions": {"ask": asks}})]
            if approved:
                flags += ["--allowedTools", *self._allow_rules(approved)]
        if self.session_id:
            flags += ["--resume", self.session_id]
        flags += ["--append-system-prompt",
                  SPOKEN_SYSTEM.format(name=self.name) + (f" Your role on this team: {self.role}" if self.role else "")]
        return flags + self._mcp_args()

    def execute(self, prompt: str, cwd: Optional[str] = None, context=None, on_text=None) -> str:
        self.last_held = []
        if not prompt or not prompt.strip():
            return "Yes, how can I help you?"
        approved, self.approve_next = self.approve_next, []

        raw = None
        if self._warm and not approved:
            if context and context.cancelled.is_set():
                return ""
            live = self._live_process(cwd)
            if live is not None:
                raw = self._execute_live(live, prompt, context=context, on_text=on_text)
                if raw is None and self.session_id is None and not (context and context.cancelled.is_set()):
                    # Nothing to lose by trying once more with a new process
                    live = self._live_process(cwd)
                    raw = self._execute_live(live, prompt, context=context, on_text=on_text) if live else None
                if raw is None and self.last_error:
                    return f"Error running {self.name}: {self.last_error[:150]}"
        if raw is None:
            # A fresh process for this turn. An approved re-run needs its own
            # command line (the approved commands are allowed for this turn only),
            # and two processes must not write to one session at once.
            if self._warm:
                self.close()
            # The prompt goes first: --mcp-config takes several values and would
            # swallow a prompt placed after it
            cmd = [self.cli_binary, "-p", prompt]
            cmd += (["--output-format", "stream-json", "--verbose", "--include-partial-messages"]
                    if on_text else ["--output-format", "json"])
            cmd += self._turn_flags(approved)
            try:
                if on_text:
                    raw = self._result_line(self.run_streaming(cmd, self._stream_handler(on_text), cwd=cwd, context=context))
                else:
                    raw = self.run_subprocess(cmd, cwd=cwd, context=context)
            finally:
                if self._warm:
                    text, session = self.parse_json_reply(raw or "", ("result",), ("session_id",))
                    self.session_id = session or self.session_id
                    self._start_live_soon()
        self.last_held = self._held(raw)
        text, session = self.parse_json_reply(raw, ("result",), ("session_id",))
        if session and session != self.session_id:
            self.session_id = session
        return text or f"{self.name} did not answer."
