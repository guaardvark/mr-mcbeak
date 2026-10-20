"""Unit tests for STT audio prep and Whisper hallucination filtering."""
import unittest
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.stt_engine import prepare_audio, is_whisper_hallucination


class TestSttPreprocess(unittest.TestCase):

    def test_peak_normalize_quiet_clip(self):
        quiet = np.full(1600, 0.004, dtype=np.float32)
        out = prepare_audio(quiet)
        self.assertAlmostEqual(float(np.max(np.abs(out))), 0.8, places=2)

    def test_flatten_2d_int16(self):
        raw = np.array([[1000], [2000], [3000]], dtype=np.int16)
        out = prepare_audio(raw)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(np.max(np.abs(out))), 0.7)

    def test_hallucination_you(self):
        self.assertTrue(is_whisper_hallucination("you"))
        self.assertTrue(is_whisper_hallucination("You."))
        self.assertTrue(is_whisper_hallucination("Thank you."))
        self.assertTrue(is_whisper_hallucination("Thanks for watching!"))
        self.assertFalse(is_whisper_hallucination("Hey Jarvis check git status"))
        self.assertFalse(is_whisper_hallucination("can you open the file"))


if __name__ == "__main__":
    unittest.main()
