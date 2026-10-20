"""
State Machine & Session Manager for Jarvis Voice Cockpit.
Manages Passive vs Active listening modes, active assistant routing, and follow-up timeouts.
"""
import time
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class StateManager:
    """
    Tracks conversational state across multi-agent turns.
    """

    def __init__(self, active_timeout_seconds: float = 30.0, default_assistant: str = "claude"):
        self.active_timeout_seconds = active_timeout_seconds
        self.default_assistant = default_assistant

        self.current_mode: str = "passive"  # "passive" or "active"
        self.active_assistant: Optional[str] = None
        self.last_interaction_time: float = 0.0
        self.turn_count: int = 0
        self.conversation_history: list[dict] = []

    def get_routing_target(self, detected_target: Optional[str]) -> Optional[str]:
        """
        Determine which assistant should receive the prompt.
        If a new wake target is explicitly detected, switches context.
        Otherwise, if within the active window, stays with the active assistant.
        """
        now = time.time()

        if detected_target:
            # Explicit target (e.g. "Hey Grok", "Hey Claude", "Jarvis")
            if detected_target == "jarvis":
                # Coordinator wake-word -> routes to active assistant or default
                target = self.active_assistant or self.default_assistant
            else:
                target = detected_target

            self.activate_assistant(target)
            return target

        # No explicit target in utterance -> Check active window
        if self.is_in_active_window():
            self.last_interaction_time = now
            return self.active_assistant

        return None

    def is_in_active_window(self) -> bool:
        """Check if we are still within the 30-second follow-up window."""
        if not self.active_assistant:
            return False
        return (time.time() - self.last_interaction_time) < self.active_timeout_seconds

    def activate_assistant(self, assistant_id: str):
        """Set the active assistant and reset the follow-up timer."""
        self.active_assistant = assistant_id
        self.current_mode = "active"
        self.last_interaction_time = time.time()
        self.turn_count += 1
        logger.info(f"Active assistant set to '{assistant_id}' (active for {self.active_timeout_seconds}s).")

    def reset_to_passive(self):
        """Revert back to passive listening mode."""
        self.current_mode = "passive"
        self.active_assistant = None
        logger.debug("State reverted to passive mode.")

    def record_turn(self, assistant_id: str, prompt: str, response: str):
        """Record interaction turn in history."""
        self.conversation_history.append({
            "timestamp": time.time(),
            "assistant": assistant_id,
            "prompt": prompt,
            "response": response
        })
