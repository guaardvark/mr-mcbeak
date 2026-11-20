"""
Group conversation ("Hey team ..."): one spoken question goes to every member,
each member sees what the others said recently, and the replies are spoken one
after another in each member's own voice.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SPOKEN_HINT = (
    "You are speaking out loud in a voice conversation. Answer in at most three "
    "short spoken sentences. Do not narrate tools, subagents, or your process.\n\n"
)
# The same hint for assistants that take it as a system prompt, so it is said
# once per conversation instead of in front of every question
SPOKEN_SYSTEM = (
    "You are {name}, speaking out loud in a voice conversation. The user's words reach you "
    "through speech recognition, so a word may be misheard (\"three plus three\" arriving as "
    "\"Street Plus 3\"); go by what they most likely meant. Answer in at most three short "
    "spoken sentences unless the message asks for something else. Plain sentences only: no "
    "lists, no markdown. Do not narrate tools, subagents, or your process."
)


def build_prompt(prompt: str, assistant_id: str, display_name: str, transcript: str,
                 participants: List[str], council: bool = False, max_sentences: int = 3,
                 tone: str = "", hint: bool = True, role: str = "") -> str:
    """Wrap the user's words with the spoken-reply hint, the shared transcript
    and, when the tone model heard something clear, a note on the tone of voice.
    hint=False for an assistant that already has the hint as its system prompt;
    role is the member's standing brief from the team file."""
    parts = []
    if council:
        others = ", ".join(p for p in participants if p != display_name) or "nobody else"
        parts.append(
            f"You are {display_name}, one voice in a spoken group conversation with the user and {others}. "
            f"Answer as yourself in at most {max_sentences} short spoken sentences; you may agree or "
            "disagree with the others by name. Do not narrate tools, subagents, or your process.\n"
        )
    elif hint:
        parts.append(SPOKEN_HINT)
    if role and (council or hint):
        parts.append(f"Your role on this team: {role}\n")
    if tone:
        parts.append(f"(Tone of voice, read from the audio by a local model and sometimes wrong: the user {tone}. "
                     "Let it shape your manner; do not mention it unless asked.)\n")
    if transcript:
        parts.append("Recent conversation (other participants; your own turns are in your session):\n"
                     f"{transcript}\n")
    parts.append(f"User: {prompt}" if (council or transcript or (role and hint)) else prompt)
    return "\n".join(parts)


def run_council(
    prompt: str,
    members: List[str],
    display_names: Dict[str, str],
    dispatch: Callable[[str, str], Tuple[str, str, int]],
    transcript_for: Callable[[str], str],
    parallel: bool = True,
    context=None,
    reply_order: str = "completion",
    on_reply: Optional[Callable[[str, str, str, int], None]] = None,
    tone: str = "",
    roles: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """
    Ask every member the same question. Return replies in playback order.
    Parallel turns snapshot prior context and default to completion order;
    sequential turns include preceding replies when transcript_for exposes them.
    """
    names = [display_names.get(m, m.title()) for m in members]

    snapshots = {m: transcript_for(m) for m in members} if parallel else {}

    def ask(member: str) -> Tuple[str, str, int]:
        full = build_prompt(
            prompt, member, display_names.get(member, member.title()),
            snapshots[member] if parallel else transcript_for(member), names, council=True, max_sentences=2,
            tone=tone, role=(roles or {}).get(member, ""),
        )
        try:
            if context and context.cancelled.is_set():
                return "", "en_US-ryan-high.onnx", 0
            return dispatch(member, full, context=context) if context else dispatch(member, full)
        except Exception as err:  # one member failing must not silence the rest
            logger.error(f"Council member {member} failed: {err}", exc_info=True)
            return f"{display_names.get(member, member)} did not answer.", "en_US-ryan-high.onnx", 0

    results: List[Tuple[str, str]] = []

    def deliver(member, result):
        if context and context.cancelled.is_set():
            return
        reply, voice, speaker = result
        results.append((member, reply))
        if on_reply:
            on_reply(member, reply, voice, speaker)

    if parallel and len(members) > 1:
        pool = ThreadPoolExecutor(max_workers=len(members))
        futures = {pool.submit(ask, m): m for m in members}
        pending = set(futures)
        try:
            ordered = list(futures)
            while pending and not (context and context.cancelled.is_set()):
                candidates = ordered[:1] if reply_order == "member" else list(pending)
                ready = [f for f in candidates if f.done()]
                if not ready:
                    from concurrent.futures import wait, FIRST_COMPLETED
                    wait(candidates, timeout=0.05, return_when=FIRST_COMPLETED)
                    continue
                for future in ready:
                    pending.remove(future)
                    ordered.remove(future)
                    deliver(futures[future], future.result())
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
    else:
        for member in members:
            if context and context.cancelled.is_set():
                break
            deliver(member, ask(member))
    return results
