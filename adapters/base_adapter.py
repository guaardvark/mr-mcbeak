"""
Abstract Base Adapter for AI Coding Assistants.

Every adapter runs a CLI in print mode, asks for JSON, keeps the session id the
CLI returns so the next turn resumes the same conversation, and honours three
per-assistant config keys:

    workspace:   directory the CLI runs in (default: where the cockpit was launched)
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
import shutil
import signal
import time

from core.privacy import child_env

logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """
    Abstract interface for invoking a CLI or API assistant.
    """

    def __init__(self, name: str, target_id: str, config: Dict[str, Any]):
        # target_id is the kind of assistant (claude, grok, ...); the name is the
        # member's own, so a team can hold several members of one kind
        self.name = config.get("name") or name
        self.target_id = target_id
        self.config = config
        # Added to every prompt, e.g. "You review changes before they land."
        self.role = str(config.get("role") or "").strip()
        # Why the last turn failed (missing CLI, not signed in, timeout), else None
        self.last_error: Optional[str] = None
        self.voice_file = config.get("voice", "en_US-libritts-high.onnx")
        self.voice_speaker = config.get("voice_speaker", 0)
        self.workspace = os.path.expanduser(config.get("workspace") or config.get("folder") or "") or None
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

    def unavailable(self) -> Optional[str]:
        """Why this member cannot answer on this machine, or None when it can."""
        binary = getattr(self, "cli_binary", None)
        if binary and not shutil.which(binary):
            return f"the {os.path.basename(binary)} command is not installed"
        return None

    def warm(self):
        """Get ready before the first question (the listener calls this)."""

    def close(self):
        """Let go of anything kept between turns."""

    def get_state(self) -> Dict[str, Any]:
        """What is needed to carry the conversation on after a restart."""
        return {"session_id": self.session_id} if self.session_id else {}

    def set_state(self, state: Dict[str, Any]):
        self.session_id = state.get("session_id") or self.session_id

    def run_subprocess(self, cmd: list[str], cwd: Optional[str] = None, timeout: Optional[int] = None, context=None) -> str:
        """
        Helper to run CLI command safely. Returns stdout, or a short spoken error.
        """
        working_dir = cwd or self.workspace or os.getcwd()
        timeout = timeout or self.timeout
        # Flags only: the user's words and long JSON arguments stay out of the log
        shown = [c for c in cmd if "\n" not in c and len(c) <= 60 and (c is cmd[0] or c.startswith("-") or len(c) <= 40)]
        logger.info(f"[{self.name}] Running: {' '.join(shown)} (cwd={working_dir})")
        self.last_error = None

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
                self.last_error = (res.stderr.strip() or res.stdout.strip() or f"exit code {res.returncode}")[:200]
                if res.stdout.strip():
                    return res.stdout.strip()
                return f"Error running {self.name}: {res.stderr.strip()[:150]}"

            return res.stdout.strip()

        except subprocess.TimeoutExpired:
            logger.error(f"[{self.name}] Command timed out after {timeout}s")
            self.last_error = f"no answer within {timeout} seconds"
            return f"Sorry, {self.name} took too long to respond."
        except Exception as e:
            logger.error(f"[{self.name}] Execution error: {e}")
            self.last_error = str(e)[:200]
            return f"Error communicating with {self.name}: {str(e)}"

    def run_streaming(self, cmd: list[str], on_line, cwd: Optional[str] = None, context=None) -> str:
        """Like run_subprocess, but hands each stdout line to on_line as it
        arrives. Returns all of stdout, or a short spoken error."""
        import queue
        import threading
        working_dir = cwd or self.workspace or os.getcwd()
        shown = [c for c in cmd if "\n" not in c and len(c) <= 60 and (c is cmd[0] or c.startswith("-") or len(c) <= 40)]
        logger.info(f"[{self.name}] Running: {' '.join(shown)} (cwd={working_dir})")
        self.last_error = None
        if context and context.cancelled.is_set():
            return ""
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    cwd=working_dir, stdin=subprocess.DEVNULL, start_new_session=True,
                                    env=child_env(), bufsize=1)
        except Exception as e:
            logger.error(f"[{self.name}] Execution error: {e}")
            self.last_error = str(e)[:200]
            return f"Error communicating with {self.name}: {str(e)}"
        lines: "queue.Queue" = queue.Queue()
        stderr_parts = []
        reader = threading.Thread(target=lambda: ([lines.put(l) for l in proc.stdout], lines.put(None)), daemon=True)
        err_reader = threading.Thread(target=lambda: stderr_parts.append(proc.stderr.read()), daemon=True)
        reader.start()
        err_reader.start()
        collected = []
        deadline = time.monotonic() + self.timeout

        def stop_process():
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                    proc.wait(timeout=0.5)
                    return
                except ProcessLookupError:
                    return
                except subprocess.TimeoutExpired:
                    continue

        while True:
            if context and context.cancelled.is_set():
                stop_process()
                return ""
            if time.monotonic() >= deadline:
                stop_process()
                logger.error(f"[{self.name}] Command timed out after {self.timeout}s")
                self.last_error = f"no answer within {self.timeout} seconds"
                return f"Sorry, {self.name} took too long to respond."
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                break
            collected.append(line)
            try:
                on_line(line)
            except Exception as e:
                logger.error(f"[{self.name}] Stream handler error: {e}")
        proc.wait()
        err_reader.join(timeout=1.0)
        if proc.returncode != 0 and not collected:
            err = "".join(stderr_parts).strip()
            logger.warning(f"[{self.name}] Exit code {proc.returncode}: {err[:300]}")
            self.last_error = err[:200] or f"exit code {proc.returncode}"
            return f"Error running {self.name}: {err[:150]}"
        if proc.returncode != 0:
            self.last_error = ("".join(stderr_parts).strip() or f"exit code {proc.returncode}")[:200]
        return "".join(collected)

    def post_json(self, url: str, payload: dict, timeout: Optional[float] = None, context=None):
        """POST to a local or LAN service without holding up the conversation: the
        request runs on its own thread, and a cancelled turn returns None at once
        while the abandoned request finishes in the background and is discarded.
        Raises what the request raised."""
        import threading
        import requests
        box = {}

        def call():
            try:
                box["response"] = requests.post(url, json=payload, timeout=timeout or self.timeout)
            except Exception as err:
                box["error"] = err
        worker = threading.Thread(target=call, name="http", daemon=True)
        worker.start()
        while worker.is_alive():
            if context and context.cancelled.is_set():
                return None
            worker.join(0.1)
        if "error" in box:
            raise box["error"]
        return box["response"]

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
