"""Cancellation and timing shared by one conversational turn."""
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class TurnContext:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    cancelled: threading.Event = field(default_factory=threading.Event)
    started_at: float = field(default_factory=time.monotonic)
    _audible: bool = False

    def cancel(self):
        self.cancelled.set()

    def timing(self, stage):
        logging.getLogger(__name__).info("turn=%s stage=%s elapsed=%.3fs", self.id, stage,
                                         time.monotonic() - self.started_at)

    def first_audio(self):
        if not self._audible:
            self._audible = True
            self.timing("first_audio")
