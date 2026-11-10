"""
Abstract Base Adapter for AI Coding Assistants.

Every adapter runs a CLI in print mode, asks for JSON, keeps the session id the
CLI returns so the next turn resumes the same conversation, and honours three
per-assistant config keys:

    workspace:   directory the CLI runs in (default: where jarvis was launched)
    model:       model override passed to the CLI (default: the CLI's own)
    permissions: "readonly" (default) or "full". Read-only keeps a stray phrase
                 picked up by an always-on microphone from editing files or
                 running shell commands.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import json
import subprocess
import logging
import os
import signal
import time

from core.privacy import child_env

logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """
    Abstract interface for invoking a CLI or API assistant.
    """

    def __init__(self, name: str, target_id: str, config: Dict[str, Any]):
        self.name = name
        self.target_id = target_id
        self.config = config
        self.voice_file = config.get("voice", "en_US-ryan-high.onnx")
        self.voice_speaker = config.get("voice_speaker", 0)
        self.workspace = os.path.expanduser(config.get("workspace") or "") or None
        self.model = config.get("model") or None
        self.readonly = str(config.get("permissions", "readonly")).lower() != "full"
        self.timeout = int(config.get("timeout_seconds", 120))
        self.session_id: Optional[str] = None

    @abstractmethod
    def execute(self, prompt: str, cwd: Optional[str] = None, context=None) -> str:
        """
        Execute a prompt synchronously and return the full text response.
        """
        pass

    @property
    def capabilities(self):
        return {"readonly": self.target_id != "guaardvark", "cancel_cli": True,
                "cancel_http": "discard_response"}

    def reset_session(self):
        """Forget the CLI conversation; the next turn starts fresh."""
        self.session_id = None

    def run_subprocess(self, cmd: list[str], cwd: Optional[str] = None, timeout: Optional[int] = None, context=None) -> str:
        """
        Helper to run CLI command safely. Returns stdout, or a short spoken error.
        """
        working_dir = cwd or self.workspace or os.getcwd()
        timeout = timeout or self.timeout
        logger.info(f"[{self.name}] Running: {' '.join(cmd[:6])}... (cwd={working_dir})")

        try:
            if context and context.cancelled.is_set():
                return ""
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, cwd=working_dir, stdin=subprocess.DEVNULL,
                                    start_new_session=True, env=child_env())
            deadline = time.monotonic() + timeout
            while True:
                cancelled = context and context.cancelled.is_set()
                if cancelled or time.monotonic() >= deadline:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.communicate(timeout=0.5)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.communicate()
                    if cancelled:
                        return ""
                    raise subprocess.TimeoutExpired(cmd, timeout)
                try:
                    stdout, stderr = proc.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            res = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
            if res.returncode != 0:
                logger.warning(f"[{self.name}] Exit code {res.returncode}: {res.stderr[:300]}")
                if res.stdout.strip():
                    return res.stdout.strip()
                return f"Error running {self.name}: {res.stderr.strip()[:150]}"

            return res.stdout.strip()

        except subprocess.TimeoutExpired:
            logger.error(f"[{self.name}] Command timed out after {timeout}s")
            return f"Sorry, {self.name} took too long to respond."
        except Exception as e:
            logger.error(f"[{self.name}] Execution error: {e}")
            return f"Error communicating with {self.name}: {str(e)}"

    @staticmethod
    def parse_json_reply(raw: str, text_keys: tuple, session_keys: tuple) -> tuple[str, Optional[str]]:
        """
        Pull (reply_text, session_id) out of a CLI's JSON output. Falls back to
        the raw text when the output is not JSON, so a CLI error still gets spoken.
        """
        blob = (raw or "").strip()
        if not blob:
            return "", None
        start = blob.find("{")
        if start == -1:
            return blob, None
        try:
            data = json.loads(blob[start:])
        except json.JSONDecodeError:
            return blob, None
        if not isinstance(data, dict):
            return blob, None
        text = ""
        for key in text_keys:
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                text = val.strip()
                break
        session = None
        for key in session_keys:
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                session = val.strip()
                break
        if data.get("is_error") or str(data.get("status", "")).upper() in ("ERROR", "FAILED"):
            logger.warning(f"CLI reported an error: {blob[:300]}")
        if not text:
            # Valid JSON with no reply (e.g. the CLI tried a denied action and gave up).
            # Never speak the JSON itself.
            logger.warning(f"CLI returned no reply text: {blob[:300]}")
            denied = data.get("denied_actions") or data.get("permission_denials")
            if denied:
                return "I tried to run something that voice mode does not allow. Ask me to answer from what I know, or set permissions to full.", session
            return "", session
        return text, session
