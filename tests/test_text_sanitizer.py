"""
Unit tests for text_sanitizer.py.
"""
import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.text_sanitizer import clean_text_for_tts, split_into_sentences, extract_speakable_text


class TestTextSanitizer(unittest.TestCase):

    def test_strip_code_blocks(self):
        raw = "Here is the code:\n```python\ndef foo():\n    return 42\n```\nLet me know if you need changes."
        cleaned = clean_text_for_tts(raw)
        self.assertNotIn("def foo()", cleaned)
        self.assertNotIn("```", cleaned)
        self.assertIn("Here is the code", cleaned)
        self.assertIn("Let me know if you need changes", cleaned)

    def test_inline_code_and_bold(self):
        raw = "You should use `requests.get()` to fetch the **user profile**."
        cleaned = clean_text_for_tts(raw)
        self.assertEqual(cleaned, "You should use requests.get() to fetch the user profile.")

    def test_markdown_links_and_headers(self):
        raw = "# System Overview\nCheck out [the documentation](https://example.com/docs) for more details."
        cleaned = clean_text_for_tts(raw)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("System Overview", cleaned)
        self.assertIn("the documentation", cleaned)

    def test_split_into_sentences(self):
        text = "Hello there! This is sentence one. And here is sentence two? That's all."
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 4)
        self.assertTrue(sentences[0].endswith("!"))
        self.assertTrue(sentences[1].endswith("."))
        self.assertTrue(sentences[2].endswith("?"))

    def test_extract_speakable_drops_agent_team_dump(self):
        raw = (
            "I'll look at the latest sessions so I can tell whether k is an acknowledgment.\n"
            "Dispatching prompt to Grok...\n"
            "[GROK] Running: grok -p k\n"
            "Heard you. The transcript came through as just k.\n"
            "If you said more than that, Whisper still clipped the utterance. "
            "If that was the whole message: I'm here — go ahead."
        )
        spoken = extract_speakable_text(raw)
        self.assertNotIn("I'll look", spoken)
        self.assertNotIn("Dispatching", spoken)
        self.assertIn("I'm here", spoken)
        self.assertLessEqual(spoken.count("."), 3)


if __name__ == "__main__":
    unittest.main()
