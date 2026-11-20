"""Speech gate, echo guard and their use in the recorder, without audio hardware."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.audio_recorder import AudioRecorder
from core.speech_gate import EchoGuard, EnergyGate, SileroGate


class EchoGuardTests(unittest.TestCase):
    def test_without_playback_level_it_is_the_fixed_level(self):
        guard = EchoGuard(base=0.015)
        guard.heard_playback(0.5)
        self.assertEqual(guard.level(), 0.015)

    def test_nothing_interrupts_until_playback_was_heard(self):
        guard = EchoGuard(0.015, 2.0, lambda: 0.1)
        self.assertEqual(guard.level(), float("inf"))
        for _ in range(10):
            guard.heard_playback(0.02)   # a fifth of what was played comes back
        self.assertAlmostEqual(guard.gain, 0.2)
        self.assertAlmostEqual(guard.level(), 0.04)

    def test_level_follows_the_reply_and_never_drops_below_base(self):
        playing = [0.1]
        guard = EchoGuard(0.015, 2.0, lambda: playing[0])
        for _ in range(10):
            guard.heard_playback(0.02)
        playing[0] = 0.3
        self.assertAlmostEqual(guard.level(), 0.12)
        playing[0] = 0.001
        self.assertEqual(guard.level(), 0.015)

    def test_a_few_loud_frames_do_not_move_the_gain(self):
        guard = EchoGuard(0.015, 2.0, lambda: 0.1)
        for n in range(40):
            guard.heard_playback(0.2 if n % 10 == 0 else 0.02)   # the user, now and then
        self.assertAlmostEqual(guard.gain, 0.2)


class RecorderTests(unittest.TestCase):
    def frames(self, recorder, level, count, start=1.0):
        audio = np.full((800, 1), level, dtype=np.float32)
        times = iter(start + 0.05 * n for n in range(count))
        with patch("core.audio_recorder.time.monotonic", side_effect=lambda: next(times)):
            for _ in range(count):
                recorder._audio_callback(audio, 800, None, None)

    def test_echo_of_the_reply_does_not_interrupt_it(self):
        guard = EchoGuard(0.015, 2.0, lambda: 0.1)
        recorder = AudioRecorder(tts_engine=Mock(is_playing=True), echo_guard=guard)
        self.frames(recorder, 0.03, 30)          # louder than the old fixed level
        self.assertFalse(recorder._interrupt.is_set())
        self.frames(recorder, 0.2, 3, start=5.0)  # the user, well over the echo
        self.assertTrue(recorder._interrupt.is_set())

    def test_gate_decides_what_is_speech(self):
        gate = Mock(return_value=False)
        recorder = AudioRecorder(tts_engine=Mock(is_playing=False), speech_gate=gate)
        self.frames(recorder, 0.2, 5)             # loud, but the gate says it is not speech
        self.assertFalse(recorder._is_speaking)
        gate.return_value = True
        self.frames(recorder, 0.004, 3, start=3.0)  # soft, and the gate says it is
        self.assertTrue(recorder._is_speaking)
        self.assertEqual(gate.call_count, 8)


class GateTests(unittest.TestCase):
    def test_energy_gate(self):
        gate = EnergyGate(0.015)
        self.assertTrue(gate(np.zeros(800), 0.02))
        self.assertFalse(gate(np.zeros(800), 0.01))

    def test_silero_calls_noise_quiet_and_carries_samples_between_frames(self):
        try:
            gate = SileroGate()
        except Exception as err:   # onnxruntime or the model not installed
            self.skipTest(str(err))
        noise = np.random.default_rng(1).normal(0, 0.05, 800).astype(np.float32)
        for _ in range(20):
            self.assertFalse(gate(noise, 0.05))
        self.assertLess(len(gate._pending), 512)
        self.assertLess(gate.last, 0.5)


if __name__ == "__main__":
    unittest.main()
