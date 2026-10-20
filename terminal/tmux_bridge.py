"""
Tmux Multi-Agent Cockpit Bridge.
Manages a multi-pane Tmux session where all 5 AI assistants run interactively.
Supports injecting voice prompts into running panes and capturing output.
"""
import shutil
import subprocess
import logging
import time
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class TmuxBridge:
    """
    Orchestrates interactive multi-agent terminal sessions via Tmux.
    """

    SESSION_NAME = "jarvis"

    # Pane mapping within the session window
    PANE_MAP = {
        "claude": 0,
        "grok": 1,
        "antigravity": 2,
        "agy": 2,
        "cursor": 3,
        "guaardvark": 4,
    }

    def __init__(self, session_name: str = "jarvis"):
        self.session_name = session_name
        self.tmux_bin = shutil.which("tmux")

    @property
    def is_available(self) -> bool:
        return self.tmux_bin is not None

    def session_exists(self) -> bool:
        if not self.is_available:
            return False
        res = subprocess.run([self.tmux_bin, "has-session", "-t", self.session_name], capture_output=True)
        return res.returncode == 0

    def create_cockpit_session(self):
        """
        Create a 5-pane split terminal layout with interactive CLIs.
        """
        if not self.is_available:
            logger.warning("tmux is not installed on system. Active terminal mode unavailable.")
            return False

        if self.session_exists():
            logger.info(f"Tmux session '{self.session_name}' already exists.")
            return True

        logger.info(f"Creating 5-pane Tmux multi-agent cockpit '{self.session_name}'...")

        # 1. Create base session (Pane 0: Claude)
        subprocess.run([self.tmux_bin, "new-session", "-d", "-s", self.session_name, "-n", "cockpit", "claude"])

        # 2. Split vertically (Pane 1: Grok)
        subprocess.run([self.tmux_bin, "split-window", "-h", "-t", f"{self.session_name}:0.0", "grok"])

        # 3. Split horizontally left bottom (Pane 2: AGY)
        subprocess.run([self.tmux_bin, "split-window", "-v", "-t", f"{self.session_name}:0.0", "agy"])

        # 4. Split horizontally right bottom (Pane 3: Cursor)
        subprocess.run([self.tmux_bin, "split-window", "-v", "-t", f"{self.session_name}:0.1", "agent"])

        # 5. Split bottom (Pane 4: Guaardvark)
        subprocess.run([self.tmux_bin, "split-window", "-v", "-t", f"{self.session_name}:0.2", "guaardvark"])

        # Select layout tiled for balanced view
        subprocess.run([self.tmux_bin, "select-layout", "-t", f"{self.session_name}:0", "tiled"])

        logger.info("Tmux multi-agent cockpit created successfully!")
        return True

    def send_prompt(self, target_id: str, prompt: str) -> bool:
        """
        Inject keystrokes / text prompt directly into the assistant's active terminal pane.
        """
        if not self.is_available or not self.session_exists():
            return False

        pane_idx = self.PANE_MAP.get(target_id.lower(), 0)
        target_spec = f"{self.session_name}:0.{pane_idx}"

        logger.info(f"Injecting prompt to Tmux pane {target_spec}: '{prompt[:50]}...'")

        # Select pane & send keys
        subprocess.run([self.tmux_bin, "select-pane", "-t", target_spec])
        # Send text followed by Enter
        subprocess.run([self.tmux_bin, "send-keys", "-t", target_spec, prompt, "C-m"])
        return True

    def capture_output(self, target_id: str, lines: int = 15) -> str:
        """
        Capture the last N lines of output from the target pane.
        """
        if not self.is_available or not self.session_exists():
            return ""

        pane_idx = self.PANE_MAP.get(target_id.lower(), 0)
        target_spec = f"{self.session_name}:0.{pane_idx}"

        res = subprocess.run(
            [self.tmux_bin, "capture-pane", "-p", "-t", target_spec, "-S", f"-{lines}"],
            capture_output=True,
            text=True
        )
        if res.returncode == 0:
            return res.stdout.strip()
        return ""
