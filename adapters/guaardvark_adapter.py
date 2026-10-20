"""
Guaardvark System Adapter.
Invokes local Guaardvark via REST API or CLI (`guaardvark ask "<prompt>"`).
"""
import shutil
import requests
import logging
from typing import Optional, Dict, Any
from adapters.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class GuaardvarkAdapter(BaseAdapter):
    """
    Adapter for local Guaardvark autonomous AI operating system.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(name="Guaardvark", target_id="guaardvark", config=config)
        self.cli_binary = shutil.which("guaardvark") or "~/.local/bin/guaardvark"
        self.api_url = config.get("api_url", "http://127.0.0.1:5001/api/chat")

    def execute(self, prompt: str, cwd: Optional[str] = None) -> str:
        if not prompt or not prompt.strip():
            return "Guaardvark online. Standing by for instructions."

        # Try REST API first (faster if backend is running)
        try:
            resp = requests.post(
                self.api_url,
                json={"message": prompt, "session_id": "jarvis_voice"},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                reply = data.get("response") or data.get("message") or data.get("text")
                if reply:
                    return reply.strip()
        except Exception as e:
            logger.debug(f"Guaardvark API offline ({e}), falling back to CLI")

        # Fallback to CLI
        cmd = [self.cli_binary, "--timeout", "10", "ask", prompt]
        return self.run_subprocess(cmd, cwd=cwd, timeout=60)
