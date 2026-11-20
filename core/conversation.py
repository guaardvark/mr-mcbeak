"""Serialize backend turns while keeping transcription and local controls responsive."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
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
        # The last question sent: (target, prompt, context, time of its last word)
        self.last_sent = None
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

    def submit(self, context, work, target=None, prompt=None, spoken_until=None):
        with self.lock:
            self.cancel()
            self.context = context
            self.pending = (target, prompt, context) if prompt else None
            if prompt:
                self.last_sent = (target, prompt, context, spoken_until or time.monotonic())
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

    def continuation_of(self, target, started_at, window):
        """The last question to `target` if speech starting at `started_at` carries it on:
        it began within `window` seconds of that question's last word and before the
        reply had been spoken in full. A natural pause longer than the end-of-speech
        silence otherwise splits one sentence into two turns, and with fast replies
        the first half is answered before the rest is said."""
        with self.lock:
            if not self.last_sent or started_at is None:
                return None
            sent_target, prompt, context, spoken_until = self.last_sent
            if sent_target != target or not 0 <= started_at - spoken_until <= window:
                return None
            finished = getattr(context, "reply_finished_at", None)
            if finished is not None and finished <= started_at:
                return None
            return prompt

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
        if not context.cancelled.is_set():
            context.reply_finished_at = time.monotonic()

    def speak_stream(self, feed, voice, speaker, context):
        """Speak sentences from a SentenceFeed as they arrive."""
        if context.cancelled.is_set():
            return
        self.tts.speak("", voice_file=voice, speaker_id=speaker, context=context,
                       sentences=feed.sentences(context))
        with self.lock:
            self.last_reply = (feed.spoken, voice, speaker)
        if not context.cancelled.is_set():
            context.reply_finished_at = time.monotonic()

    # What people actually say for each control, after "please", "okay" and
    # the like are dropped from either end
    CONTROLS = {
        "stop": ("stop", "stop it", "stop talking", "be quiet", "quiet", "that's enough", "enough", "shush"),
        "go to sleep": ("go to sleep", "sleep", "go back to sleep", "stop listening"),
        "new conversation": ("new conversation", "start a new conversation", "start over", "fresh start",
                             "forget this conversation"),
        "repeat that": ("repeat that", "repeat", "say that again", "say it again", "come again",
                        "what did you say", "can you repeat that", "could you repeat that"),
    }
    _FILLER = ("please", "okay", "ok", "now", "thanks", "thank you", "just", "hey")

    @classmethod
    def control_command(cls, prompt):
        """The control a short phrase asks for ("stop please" -> "stop"), or None."""
        words = "".join(c if c.isalnum() or c in "' " else " " for c in (prompt or "").lower()).split()
        if not words or len(words) > 7:
            return None
        changed = True
        while changed and words:
            changed = False
            for filler in cls._FILLER:
                part = filler.split()
                if words[:len(part)] == part and len(words) > len(part):
                    words, changed = words[len(part):], True
                if words[-len(part):] == part and len(words) > len(part):
                    words, changed = words[:-len(part)], True
        said = " ".join(words)
        for command, phrases in cls.CONTROLS.items():
            if said in phrases:
                return command
        return None

    def control(self, prompt, target):
        command = self.control_command(prompt)
        if command is None:
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
