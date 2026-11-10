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
        self.busy = False
        self.turn_count: int = 0
        self.conversation_history: list[dict] = []

    def get_routing_target(self, detected_target: Optional[str]) -> Optional[str]:
        """
        Determine which assistant should receive the prompt.
        If a new wake target is explicitly detected, switches context.
        Otherwise, if within the active window, stays with the active assistant.
        """
        now = time.monotonic()

        if detected_target:
            # Explicit target (e.g. "Hey Grok", "Hey Claude", "Jarvis")
            if detected_target == "jarvis":
                # Coordinator wake-word -> routes to active assistant or default
                target = self.active_assistant or self.default_assistant
                if target == "team":
                    target = self.default_assistant
            else:
                # Includes "team": the council stays active for follow-ups
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
        return self.busy or (time.monotonic() - self.last_interaction_time) < self.active_timeout_seconds

    def activate_assistant(self, assistant_id: str):
        """Set the active assistant and reset the follow-up timer."""
        self.active_assistant = assistant_id
        self.current_mode = "active"
        self.last_interaction_time = time.monotonic()
        self.turn_count += 1
        logger.info(f"Active assistant set to '{assistant_id}' (active for {self.active_timeout_seconds}s).")

    def begin_turn(self):
        self.busy = True

    def finish_turn(self):
        self.busy = False
        self.last_interaction_time = time.monotonic()

    def reset_to_passive(self):
        """Revert back to passive listening mode."""
        self.current_mode = "passive"
        self.active_assistant = None
        logger.debug("State reverted to passive mode.")

    def record_turn(self, assistant_id: str, prompt: str, response: str):
        """Record interaction turn in history (the shared transcript)."""
        self.conversation_history.append({
            "timestamp": time.time(),
            "assistant": assistant_id,
            "prompt": prompt,
            "response": response
        })
        del self.conversation_history[:-100]

    def recent_transcript(self, limit: int = 6, exclude_assistant: Optional[str] = None,
                          names: Optional[Dict[str, str]] = None) -> str:
        """
        The last turns as spoken lines, for an assistant that was not part of
        them. Its own turns are left out because its CLI session already has them.
        """
        names = names or {}
        lines = []
        last_prompt = None
        for turn in self.conversation_history[-limit:]:
            if exclude_assistant and turn["assistant"] == exclude_assistant:
                continue
            who = names.get(turn["assistant"], turn["assistant"].title())
            reply = " ".join(str(turn["response"]).split())[:400]
            if turn["prompt"] != last_prompt:  # one user line per question, then every reply to it
                lines.append(f"User: {turn['prompt']}")
                last_prompt = turn["prompt"]
            lines.append(f"{who}: {reply}")
        return "\n".join(lines)
