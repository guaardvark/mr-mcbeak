"""
OpenAI Codex CLI Adapter.
Runs `codex exec --json`, resuming the same thread on later turns with
`codex exec resume <thread>`.

Telemetry: Codex sends usage analytics to OpenAI by default. Traced
2026-11-10 (codex-cli 0.167.0): without the overrides below a one-word turn
also contacted ab.chatgpt.com and an oaiusercontent.com host; with them, only
chatgpt.com (the model itself).
"""
import json
import shutil
from typing import Any, Dict, Optional

from adapters.base_adapter import BaseAdapter

NO_TELEMETRY_ARGS = [
    "-c", "analytics.enabled=false", "-c", "feedback.enabled=false",
    "-c", 'otel.exporter="none"', "-c", 'otel.trace_exporter="none"', "-c", 'otel.metrics_exporter="none"',
]


class CodexAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(name=config.get("name") or "Codex", target_id="codex", config=config)
        self.cli_binary = shutil.which(config.get("cli_cmd") or "codex") or "codex"
        # Arguments for an interactive `codex` in a tmux pane
        self.launch_args = list(NO_TELEMETRY_ARGS)

    @property
    def sandbox(self) -> str:
        # Codex enforces this itself: read-only, or writes inside the workspace only
        return "read-only" if self.readonly else "workspace-write"

    def execute(self, prompt: str, cwd: Optional[str] = None, context=None) -> str:
        if not prompt or not prompt.strip():
            return "Codex here. What do you need?"
        cmd = [self.cli_binary, "exec"]
        if self.session_id:
            cmd += ["resume", self.session_id]
        cmd += ["--json", "--skip-git-repo-check", *NO_TELEMETRY_ARGS,
                "-c", f'sandbox_mode="{self.sandbox}"', "-c", 'approval_policy="never"']
        if self.model:
            cmd += ["-m", self.model]
        cmd.append(prompt)
        raw = self.run_subprocess(cmd, cwd=cwd, context=context)
        text, thread = self._parse(raw)
        if thread:
            self.session_id = thread
        return text or (raw.strip()[:200] if raw and not raw.lstrip().startswith("{") else f"{self.name} did not answer.")

    @staticmethod
    def _parse(raw: str):
        """(last agent message, thread id) from codex exec --json output."""
        text, thread = "", None
        for line in (raw or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "thread.started":
                thread = event.get("thread_id") or thread
            item = event.get("item") or {}
            if event.get("type") == "item.completed" and item.get("type") == "agent_message" and item.get("text"):
                text = item["text"].strip()
        return text, thread
