"""
Multi-Target Wake Word Engine with Phonetic Fuzzy Matching.
Ported and extended from Guaardvark's wakeWordMatcher.js.

Handles:
- Multiple assistants (Claude, Grok, Antigravity, Cursor, Guaardvark, Jarvis)
- Prefix variations ("Hey", "OK", "Yo", "Hello", "Hi") and bare names
- Phonetic mis-transcription variants
- Prompt remainder extraction
- Fuzzy sliding-window Levenshtein matching
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Set, Tuple
import re


@dataclass
class MatchResult:
    detected: bool
    target: Optional[str]  # e.g., "claude", "grok", "agy", "cursor", "guaardvark", "jarvis"
    remainder: str         # The prompt payload after the wake phrase
    matched_phrase: Optional[str] = None
    similarity: float = 1.0


def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def fuzzy_index_of(text: str, phrase: str, threshold: float = 0.72) -> Tuple[int, int, float]:
    """
    Search for phrase inside text with sliding window Levenshtein similarity.
    Returns (start_idx, end_idx, best_similarity).
    """
    phrase_len = len(phrase)
    if phrase_len == 0 or len(text) < phrase_len:
        return -1, -1, 0.0

    best_start = -1
    best_end = -1
    best_sim = 0.0

    # Slide windows of length phrase_len +/- 2 chars
    for window_size in range(max(1, phrase_len - 2), min(len(text) + 1, phrase_len + 3)):
        for i in range(0, len(text) - window_size + 1):
            window = text[i : i + window_size]
            dist = levenshtein_distance(window, phrase)
            sim = 1.0 - (dist / max(len(window), phrase_len))
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                best_start = i
                best_end = i + window_size

    return best_start, best_end, best_sim


class MultiTargetWakeWordMatcher:
    """
    Matches speech transcription against registered assistants and extracts prompt payload.
    """

    DEFAULT_PREFIXES = ["hey", "ok", "yo", "hello", "hi", "listen", "dear"]

    COUNCIL_TARGET = "team"

    def __init__(self, config_assistants: Dict[str, dict] = None, coordinator_wake_words: List[str] = None,
                 council_wake_words: List[str] = None):
        self.targets: Dict[str, List[str]] = {}
        self.all_phrases: List[Tuple[str, str]] = []  # (phrase, target_id)

        if config_assistants:
            self._load_from_config(config_assistants, coordinator_wake_words or ["jarvis"], council_wake_words)

    def _load_from_config(self, assistants: Dict[str, dict], coordinator_words: List[str],
                          council_words: Optional[List[str]] = None):
        """Build phrase lookup tables from config."""
        for target_id, info in assistants.items():
            words = list(info.get("wake_words", []))
            phonetics = list(info.get("phonetic_variants", []))
            self.register_target(target_id, words, phonetics)

        # Coordinator ("jarvis") routes to one assistant; council ("team") asks them all
        coord_words = [w for w in (coordinator_words or ["jarvis", "hey jarvis"]) if "team" not in w]
        self.register_target("jarvis", coord_words, ["jarviss", "jarvises"])
        if council_words:
            self.register_target(self.COUNCIL_TARGET, council_words, ["teem"])

    def register_target(self, target_id: str, wake_words: List[str], phonetic_variants: List[str] = None):
        """Register a target with its wake words and phonetic variants."""
        variants: Set[str] = set()
        
        for w in wake_words:
            w_clean = w.lower().strip()
            variants.add(w_clean)
            # Add prefix combinations if bare
            if " " not in w_clean:
                for prefix in self.DEFAULT_PREFIXES:
                    variants.add(f"{prefix} {w_clean}")

        if phonetic_variants:
            for p in phonetic_variants:
                p_clean = p.lower().strip()
                variants.add(p_clean)
                if " " not in p_clean:
                    for prefix in self.DEFAULT_PREFIXES:
                        variants.add(f"{prefix} {p_clean}")

        # Store sorted descending by length to match longest/most specific first
        for phrase in variants:
            self.all_phrases.append((phrase, target_id))

        self.all_phrases.sort(key=lambda x: len(x[0]), reverse=True)

    # A wake phrase must begin within the first N words of the utterance, so a
    # name mentioned mid-sentence ("deploy this to the cloud") does not re-route.
    MAX_START_WORD = 2

    def match(self, raw_transcription: str, active_target: Optional[str] = None) -> MatchResult:
        """
        Check if transcription starts with a wake word (optionally after a short
        lead-in such as "okay so"). The earliest phrase wins; a longer phrase wins
        a tie at the same position. If no wake-word is found and active_target is
        set, the whole utterance is routed to active_target.
        """
        if not raw_transcription or not raw_transcription.strip():
            return MatchResult(detected=False, target=None, remainder="")

        text = raw_transcription.lower()
        text_clean = re.sub(r'[,.:;!?"\'()\[\]{}]', ' ', text)
        text_clean = re.sub(r'\s+', ' ', text_clean).strip()

        # Character offsets at which a word may start (first MAX_START_WORD+1 words)
        word_starts = [0] + [m.end() for m in re.finditer(r' ', text_clean)]
        allowed_starts = set(word_starts[: self.MAX_START_WORD + 1])

        # Phase 1: exact match at an allowed word start; earliest, then longest
        best = None
        for phrase, target_id in self.all_phrases:
            for start in sorted(allowed_starts):
                if text_clean.startswith(phrase, start):
                    end = start + len(phrase)
                    if end < len(text_clean) and text_clean[end] != ' ':
                        continue
                    if best is None or start < best[0] or (start == best[0] and len(phrase) > len(best[1])):
                        best = (start, phrase, target_id)
                    break
        if best:
            start, phrase, target_id = best
            return MatchResult(
                detected=True,
                target=target_id,
                remainder=text_clean[start + len(phrase):].strip(),
                matched_phrase=phrase,
                similarity=1.0
            )

        # Phase 2: fuzzy Levenshtein match (phrases >= 4 chars), same anchoring
        best_match = None
        best_key = None
        for phrase, target_id in self.all_phrases:
            if len(phrase) < 4:
                continue
            start, end, sim = fuzzy_index_of(text_clean, phrase, threshold=0.72)
            if start == -1 or start not in allowed_starts:
                continue
            if end < len(text_clean) and text_clean[end] != " ":
                continue
            key = (start, -sim)
            if best_key is None or key < best_key:
                best_key = key
                best_match = MatchResult(
                    detected=True,
                    target=target_id,
                    remainder=text_clean[end:].strip(),
                    matched_phrase=phrase,
                    similarity=sim
                )
        if best_match:
            return best_match

        # Phase 3: active follow-up routing
        if active_target:
            return MatchResult(
                detected=True,
                target=active_target,
                remainder=raw_transcription.strip(),
                matched_phrase="[active_followup]",
                similarity=1.0
            )

        return MatchResult(detected=False, target=None, remainder=raw_transcription.strip())
