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


def build_prompt(prompt: str, assistant_id: str, display_name: str, transcript: str,
                 participants: List[str], council: bool = False, max_sentences: int = 3) -> str:
    """Wrap the user's words with the spoken-reply hint and the shared transcript."""
    parts = []
    if council:
        others = ", ".join(p for p in participants if p != display_name) or "nobody else"
        parts.append(
            f"You are {display_name}, one voice in a spoken group conversation with the user and {others}. "
            f"Answer as yourself in at most {max_sentences} short spoken sentences; you may agree or "
            "disagree with the others by name. Do not narrate tools, subagents, or your process.\n"
        )
    else:
        parts.append(SPOKEN_HINT)
    if transcript:
        parts.append("Recent conversation (other participants; your own turns are in your session):\n"
                     f"{transcript}\n")
    parts.append(f"User: {prompt}" if (council or transcript) else prompt)
    return "\n".join(parts)


def run_council(
    prompt: str,
    members: List[str],
    display_names: Dict[str, str],
    dispatch: Callable[[str, str], Tuple[str, str, int]],
    transcript_for: Callable[[str], str],
    parallel: bool = True,
    on_reply: Optional[Callable[[str, str, str, int], None]] = None,
) -> List[Tuple[str, str]]:
    """
    Ask every member the same question. Returns [(member_id, reply)] in member
    order. on_reply(member_id, reply, voice_file, speaker_id) is called for each
    reply as soon as it can be spoken in order.
    """
    names = [display_names.get(m, m.title()) for m in members]

    def ask(member: str) -> Tuple[str, str, int]:
        full = build_prompt(
            prompt, member, display_names.get(member, member.title()),
            transcript_for(member), names, council=True, max_sentences=2,
        )
        try:
            return dispatch(member, full)
        except Exception as err:  # one member failing must not silence the rest
            logger.error(f"Council member {member} failed: {err}", exc_info=True)
            return f"{display_names.get(member, member)} did not answer.", "en_US-ryan-high.onnx", 0

    results: List[Tuple[str, str]] = []
    if parallel and len(members) > 1:
        with ThreadPoolExecutor(max_workers=len(members)) as pool:
            futures = [(m, pool.submit(ask, m)) for m in members]
            for member, fut in futures:  # speak in member order, as each finishes
                reply, voice, speaker = fut.result()
                results.append((member, reply))
                if on_reply:
                    on_reply(member, reply, voice, speaker)
    else:
        for member in members:
            reply, voice, speaker = ask(member)
            results.append((member, reply))
            if on_reply:
                on_reply(member, reply, voice, speaker)
    return results
