"""
Speak a reply while it is still being written.

The assistant's text arrives in small pieces. Each finished sentence goes
through the same cleanup as a whole reply (extract_speakable_text: no code, no
trace or "Let me check ..." lines, at most a few sentences) and is handed to
the speech engine at once, so the first sentence plays while later ones are
still being written. Measured 2026-11-10: about 0.9 s sooner on plain answers.
"""
import re
import threading
from typing import Iterator, List, Optional

from core.text_sanitizer import _META_PREFIXES, _TRACE_LINE, clean_text_for_tts

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Enough of a line to tell a trace or "Let me ..." line from an answer
_DECIDE_AT = 40


class SentenceFeed:
    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences
        self._lock = threading.Condition()
        self._out: List[str] = []
        self._line = ""          # text of the line being written
        self._used = 0           # characters of _line already spoken
        self._keep: Optional[bool] = None  # whether _line is speakable, once known
        self._in_code = False
        self._done = False
        self._muted = False

    # -- writer side -------------------------------------------------------
    def add(self, delta: str) -> bool:
        """Take more text; True when at least one sentence is ready to speak."""
        with self._lock:
            parts = delta.split("\n")
            for i, part in enumerate(parts):
                self._line += part
                if i < len(parts) - 1:
                    self._finish_line()
                else:
                    self._advance(final=False)
            self._lock.notify_all()
            return bool(self._out)

    def close(self):
        with self._lock:
            self._finish_line()
            self._done = True
            self._lock.notify_all()

    def mute(self):
        """Speak nothing more (the turn was held for approval)."""
        with self._lock:
            self._muted = self._done = True
            self._lock.notify_all()

    @property
    def spoken(self) -> str:
        with self._lock:
            return " ".join(self._out)

    def _finish_line(self):
        self._advance(final=True)
        self._line, self._used, self._keep = "", 0, None

    def _advance(self, final: bool):
        stripped = self._line.strip()
        if stripped.startswith("```"):
            if final:
                self._in_code = not self._in_code
            self._keep = False
            return
        if self._in_code:
            self._keep = False
            return
        if self._keep is None and (final or len(stripped) >= _DECIDE_AT or _SENTENCE_END.search(self._line)):
            lower = stripped.lower()
            self._keep = bool(stripped) and not _TRACE_LINE.match(stripped) and \
                not any(lower.startswith(p) for p in _META_PREFIXES)
        if not self._keep:
            return
        pending = self._line[self._used:]
        pieces = _SENTENCE_END.split(pending)
        complete = pieces if final else pieces[:-1]
        for sentence in complete:
            self._emit(sentence)
        if not final and len(pieces) > 1:
            self._used = len(self._line) - len(pieces[-1])
        elif final:
            self._used = len(self._line)

    def _emit(self, sentence: str):
        if self._muted or len(self._out) >= self.max_sentences:
            return
        text = re.sub(r"https?://\S+", "", sentence)
        text = re.sub(r"/home/\S+", "", text)
        cleaned = re.sub(r":\.$", ".", clean_text_for_tts(text))
        if cleaned and len(cleaned) > 1 and re.search(r"[A-Za-z0-9]", cleaned):
            self._out.append(cleaned)

    # -- speaker side ------------------------------------------------------
    def sentences(self, context=None) -> Iterator[str]:
        """Sentences as they become ready; ends when the reply is complete,
        muted, or the turn is cancelled."""
        i = 0
        while True:
            with self._lock:
                while i >= len(self._out) and not self._done and not self._muted:
                    if context is not None and context.cancelled.is_set():
                        return
                    self._lock.wait(0.1)
                if self._muted or (context is not None and context.cancelled.is_set()):
                    return
                if i >= len(self._out):
                    return
                sentence = self._out[i]
                i += 1
            yield sentence
