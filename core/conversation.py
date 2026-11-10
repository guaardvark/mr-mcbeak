"""Serialize backend turns while keeping transcription and local controls responsive."""
from concurrent.futures import ThreadPoolExecutor
import threading
from core.turn_context import TurnContext


class Conversation:
    def __init__(self, state, tts, dispatcher):
        self.state, self.tts, self.dispatcher = state, tts, dispatcher
        self.on_cancel = None
        self.on_reset = None
        self.context = None
        self.future = None
        self.last_reply = None
        # (target, prompt, context) of the turn still waiting for its reply, so
        # words added while an assistant is thinking join that question instead
        # of cancelling it
        self.pending = None
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turn")

    def cancel(self):
        with self.lock:
            if self.context:
                self.context.cancel()
            if self.future:
                self.future.cancel()
            if self.on_cancel:
                self.on_cancel()
            self.tts.stop()
            self.state.finish_turn()

    def submit(self, context, work, target=None, prompt=None):
        with self.lock:
            self.cancel()
            self.context = context
            self.pending = (target, prompt, context) if prompt else None
            self.state.begin_turn()

            def run():
                try:
                    if not context.cancelled.is_set():
                        work(context)
                finally:
                    with self.lock:
                        if self.context is context:
                            self.state.finish_turn()
                        self._answered(context)
            self.future = self.executor.submit(run)

    def unanswered(self, target):
        """The prompt of the turn to `target` still waiting for its reply, or None."""
        with self.lock:
            if self.pending and self.pending[0] == target and not self.pending[2].cancelled.is_set():
                return self.pending[1]
            return None

    def _answered(self, context):
        if self.pending and self.pending[2] is context:
            self.pending = None

    def record_turn(self, member, prompt, reply, context):
        with self.lock:
            if context.cancelled.is_set():
                return False
            self._answered(context)
            self.state.record_turn(member, prompt, reply)
            return True

    def speak(self, text, voice, speaker, context):
        with self.lock:
            if context.cancelled.is_set():
                return
            self.last_reply = (text, voice, speaker)
        self.tts.speak(text, voice_file=voice, speaker_id=speaker, context=context)

    def control(self, prompt, target):
        command = prompt.lower().strip(" .!?\t\n")
        if command not in {"stop", "go to sleep", "new conversation", "repeat that"}:
            return False
        self.cancel()
        if command == "go to sleep":
            self.state.reset_to_passive()
        elif command == "new conversation":
            self.state.conversation_history.clear()
            # Reset after any cancelled backend finishes, before the next turn.
            def reset():
                self.dispatcher.reset_sessions(target)
                if self.on_reset:
                    self.on_reset(target)
            self.executor.submit(reset)
            self.last_reply = None
        elif command == "repeat that" and self.last_reply:
            reply = self.last_reply
            self.submit(TurnContext(), lambda ctx: self.speak(*reply, ctx))
        return True

    def close(self):
        self.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
