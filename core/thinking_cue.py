"""A soft pulse while an assistant is working, so a long wait does not sound like a dead mic."""
import threading
from types import SimpleNamespace


def start_thinking_cue(tts, context, every: float = 3.0):
    """Pulse first after half an interval, then every `every` seconds, until the
    returned function is called or the turn is cancelled. Returns the stop function.

    The pulse is played with its own cancellation context, so once stopped it can
    neither start nor play after a reply that took the speaker first.
    """
    stopped = threading.Event()
    cue_context = SimpleNamespace(cancelled=stopped)

    def loop():
        delay = every / 2
        while not stopped.wait(delay):
            if context.cancelled.is_set():
                return
            tts.play_sound_cue("thinking", context=cue_context)
            delay = every

    threading.Thread(target=loop, daemon=True, name="thinking-cue").start()
    return stopped.set
