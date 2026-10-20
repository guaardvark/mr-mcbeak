"""
Abstract Base Adapter for AI Coding Assistants.
"""
from abc import ABC, abstractmethod
from typing import Generator, Optional, Dict, Any
import subprocess
import logging
import os

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

    @abstractmethod
    def execute(self, prompt: str, cwd: Optional[str] = None) -> str:
        """
        Execute a prompt synchronously and return the full text response.
        """
        pass

    def run_subprocess(self, cmd: list[str], cwd: Optional[str] = None, timeout: int = 120) -> str:
        """
        Helper to run CLI command safely.
        """
        working_dir = cwd or os.getcwd()
        logger.info(f"[{self.name}] Running: {' '.join(cmd[:4])}... (cwd={working_dir})")

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=working_dir,
                timeout=timeout
            )
            if res.returncode != 0:
                logger.warning(f"[{self.name}] Exit code {res.returncode}: {res.stderr[:200]}")
                if res.stdout:
                    return res.stdout.strip()
                return f"Error running {self.name}: {res.stderr.strip()[:150]}"

            return res.stdout.strip()

        except subprocess.TimeoutExpired:
            logger.error(f"[{self.name}] Command timed out after {timeout}s")
            return f"Sorry, {self.name} took too long to respond."
        except Exception as e:
            logger.error(f"[{self.name}] Execution error: {e}")
            return f"Error communicating with {self.name}: {str(e)}"
