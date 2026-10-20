"""
Unit tests for StateManager and AdapterDispatcher.
"""
import unittest
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.state_manager import StateManager
from adapters.dispatcher import AdapterDispatcher
from adapters.grok_adapter import GrokAdapter


class TestStateManager(unittest.TestCase):

    def test_state_transitions(self):
        sm = StateManager(active_timeout_seconds=2.0, default_assistant="claude")
        self.assertEqual(sm.current_mode, "passive")
        self.assertIsNone(sm.active_assistant)

        # Activate via wake-word
        target = sm.get_routing_target("grok")
        self.assertEqual(target, "grok")
        self.assertEqual(sm.current_mode, "active")
        self.assertEqual(sm.active_assistant, "grok")

        # Immediate follow-up (no wake word)
        followup = sm.get_routing_target(None)
        self.assertEqual(followup, "grok")

        # Explicit switch to Claude
        switch_target = sm.get_routing_target("claude")
        self.assertEqual(switch_target, "claude")
        self.assertEqual(sm.active_assistant, "claude")

        # Wait for timeout (2s)
        time.sleep(2.1)
        self.assertFalse(sm.is_in_active_window())
        timeout_target = sm.get_routing_target(None)
        self.assertIsNone(timeout_target)


class TestDispatcher(unittest.TestCase):

    def test_dispatcher_registration(self):
        assistants_cfg = {
            "claude": {"target_id": "claude", "voice": "en_US-ryan-high.onnx", "voice_speaker": 0},
            "grok": {"target_id": "grok", "voice": "en_US-ryan-high.onnx", "voice_speaker": 0},
            "antigravity": {"target_id": "agy", "voice": "en_US-libritts-high.onnx", "voice_speaker": 12},
            "cursor": {"target_id": "cursor", "voice": "en_US-libritts-high.onnx", "voice_speaker": 24},
            "guaardvark": {"target_id": "guaardvark", "voice": "en_US-libritts-high.onnx", "voice_speaker": 42},
        }
        dispatcher = AdapterDispatcher(assistants_cfg)
        self.assertIn("claude", dispatcher.adapters)
        self.assertIn("grok", dispatcher.adapters)
        self.assertIn("agy", dispatcher.adapters)
        self.assertIn("cursor", dispatcher.adapters)
        self.assertIn("guaardvark", dispatcher.adapters)


class TestGrokAdapter(unittest.TestCase):

    def test_final_text_from_json_result(self):
        raw = '{"type":"result","result":"Four words, as asked."}'
        self.assertEqual(GrokAdapter._final_text(raw), "Four words, as asked.")

    def test_final_text_fallback_plain(self):
        self.assertEqual(GrokAdapter._final_text("just a sentence"), "just a sentence")


if __name__ == "__main__":
    unittest.main()
