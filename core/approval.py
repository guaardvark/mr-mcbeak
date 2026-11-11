"""
Spoken approval for actions an assistant was held back from.

An assistant with full permissions runs in a mode where ordinary work goes
ahead and risky actions are refused and reported (for Claude Code: auto mode,
plus the "ask first" rules in claude_adapter.CONFIRM_FIRST). The refusal comes
back as a list of held actions; the cockpit asks out loud whether to go ahead,
and a spoken yes re-runs the same session with exactly those actions allowed.
"""
import os
import re
import time
from typing import Dict, List, Optional, Tuple

YES = ("yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go ahead", "do it", "go for it",
       "approved", "approve", "confirm", "confirmed", "please do", "affirmative", "absolutely",
       "of course", "that's fine", "fine", "proceed")
NO = ("no", "nope", "don't", "do not", "cancel", "never mind", "nevermind", "wait", "hold on",
      "not now", "negative", "not yet", "leave it")
# How long a question stays open
OPEN_SECONDS = 90.0
_SPLIT = re.compile(r"&&|\|\||;|\|")


def answer(text: str) -> Optional[bool]:
    """True for a spoken yes, False for a no, None for anything else (a new request)."""
    t = re.sub(r"[^a-z' ]", " ", (text or "").lower())
    t = " ".join(t.split())
    if not t or len(t.split()) > 8:
        return None
    if any(t == n or t.startswith(n + " ") for n in NO):
        return False
    if any(t == y or t.startswith(y + " ") for y in YES):
        return True
    return None


def subcommands(command: str) -> List[str]:
    return [s.strip() for s in _SPLIT.split(command or "") if s.strip()]


def rule_prefix(rule: str) -> Optional[str]:
    """'Bash(git push:*)' -> 'git push'; None for other rules."""
    m = re.fullmatch(r"Bash\((.+?)(?::\*)?\)", rule.strip())
    return m.group(1).strip() if m else None


def describe(held: List[dict], rules: List[str]) -> str:
    """What to say: the risky part of each held action, e.g. "run rm notes.txt"."""
    prefixes = [p for p in (rule_prefix(r) for r in rules) if p]
    parts = []
    for h in held:
        if h["tool"] == "Bash":
            subs = subcommands(h["detail"])
            risky = [s for s in subs if any(s.startswith(p) for p in prefixes)] or subs[:1]
            parts += [f"run {s}" for s in risky]
        elif h["detail"]:
            verb = {"Write": "write", "Edit": "edit", "MultiEdit": "edit", "NotebookEdit": "edit"}.get(h["tool"], "use " + h["tool"] + " on")
            parts.append(f"{verb} {os.path.basename(h['detail'].rstrip('/')) or h['detail']}")
        else:
            parts.append(f"use {h['tool']}")
    text = ", then ".join(dict.fromkeys(parts))
    return text if len(text) <= 160 else text[:157].rsplit(" ", 1)[0] + "..."


class HeldActions:
    """Open questions, one per assistant."""

    def __init__(self):
        self._open: Dict[str, Tuple[float, List[dict]]] = {}

    def hold(self, target: str, held: List[dict]):
        self._open[target] = (time.monotonic(), held)

    def take(self, target: str) -> Optional[List[dict]]:
        """The open question for target, removed; None when there is none or it expired."""
        item = self._open.pop(target, None)
        if not item or time.monotonic() - item[0] > OPEN_SECONDS:
            return None
        return item[1]

    def pending(self, target: str) -> bool:
        item = self._open.get(target)
        return bool(item) and time.monotonic() - item[0] <= OPEN_SECONDS

    def drop(self, target: str):
        self._open.pop(target, None)
