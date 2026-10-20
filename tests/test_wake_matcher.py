"""
Unit tests for MultiTargetWakeWordMatcher.
"""
import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.wake_word_matcher import MultiTargetWakeWordMatcher, MatchResult


class TestWakeWordMatcher(unittest.TestCase):

    def setUp(self):
        assistants = {
            "claude": {
                "wake_words": ["claude", "hey claude", "ok claude"],
                "phonetic_variants": ["clod", "clawed", "claud"]
            },
            "grok": {
                "wake_words": ["grok", "hey grok", "ok grok"],
                "phonetic_variants": ["grock", "croc", "brock", "grut", "groot"]
            },
            "antigravity": {
                "wake_words": ["antigravity", "anti gravity", "agy", "hey antigravity", "hey agy"],
                "phonetic_variants": ["gravity"]
            },
            "cursor": {
                "wake_words": ["cursor", "hey cursor"],
                "phonetic_variants": ["curser", "cursir"]
            },
            "guaardvark": {
                "wake_words": ["guaardvark", "hey guaardvark"],
                "phonetic_variants": ["guard vark", "guardvark", "guad vark", "godvark"]
            }
        }
        self.matcher = MultiTargetWakeWordMatcher(
            assistants, coordinator_wake_words=["jarvis", "hey jarvis"], council_wake_words=["team", "hey team"]
        )

    def test_exact_wake_word_extraction(self):
        res = self.matcher.match("Hey Claude, please check the git status")
        self.assertTrue(res.detected)
        self.assertEqual(res.target, "claude")
        self.assertEqual(res.remainder, "please check the git status")

    def test_phonetic_variant_grok(self):
        res = self.matcher.match("yo grock write a python test")
        self.assertTrue(res.detected)
        self.assertEqual(res.target, "grok")
        self.assertEqual(res.remainder, "write a python test")

        res = self.matcher.match("Okay, grut.")
        self.assertTrue(res.detected)
        self.assertEqual(res.target, "grok")

    def test_antigravity_agy_acronym(self):
        res = self.matcher.match("hey agy explain this codebase")
        self.assertTrue(res.detected)
        self.assertEqual(res.target, "antigravity")
        self.assertEqual(res.remainder, "explain this codebase")

    def test_cursor_fuzzy(self):
        res = self.matcher.match("ok curser find the bug in login")
        self.assertTrue(res.detected)
        self.assertEqual(res.target, "cursor")
        self.assertEqual(res.remainder, "find the bug in login")

    def test_guaardvark_fuzzy_guard_vark(self):
        res = self.matcher.match("hey guard vark what is system health")
        self.assertTrue(res.detected)
        self.assertEqual(res.target, "guaardvark")
        self.assertEqual(res.remainder, "what is system health")

    def test_coordinator_jarvis(self):
        res = self.matcher.match("hey jarvis give me a system summary")
        self.assertTrue(res.detected)
        self.assertEqual(res.target, "jarvis")
        self.assertEqual(res.remainder, "give me a system summary")

    def test_active_followup_routing(self):
        # When active_target is provided and user doesn't say a wake word
        res = self.matcher.match("now run the unit tests", active_target="claude")
        self.assertTrue(res.detected)
        self.assertEqual(res.target, "claude")
        self.assertEqual(res.remainder, "now run the unit tests")

    def test_no_wake_word_passive(self):
        res = self.matcher.match("the weather is nice today")
        self.assertFalse(res.detected)
        self.assertIsNone(res.target)

    def test_croc_does_not_match_inside_crack(self):
        res = self.matcher.match("a crack.")
        self.assertFalse(res.detected)

    def test_name_mid_sentence_does_not_reroute(self):
        res = self.matcher.match("grok deploy this to the cloud")
        self.assertEqual(res.target, "grok")
        self.assertEqual(res.remainder, "deploy this to the cloud")
        res = self.matcher.match("tell me about the cursor position in vim")
        self.assertFalse(res.detected)

    def test_short_lead_in_allowed(self):
        res = self.matcher.match("okay so hey claude run it")
        self.assertEqual(res.target, "claude")
        self.assertEqual(res.remainder, "run it")

    def test_team_wake_word_routes_to_council(self):
        res = self.matcher.match("hey team which database should we use")
        self.assertEqual(res.target, "team")
        self.assertEqual(res.remainder, "which database should we use")


if __name__ == "__main__":
    unittest.main()
