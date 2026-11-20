"""
Is this 50 ms of microphone audio someone speaking?

Two answers. EnergyGate is the original: louder than a fixed level. It hears a
door, a fan or typing as speech and misses soft speech below the level.
SileroGate asks Silero VAD (the model faster-whisper already ships and uses on
finished clips; about 0.04 ms per 32 ms window on one CPU thread), which
answers by what the sound is rather than how loud it is.

A gate is called once for every frame the microphone delivers, in order, so
the model's memory of the last frames stays true.
"""
import logging
import os
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

WINDOW = 512     # samples per model step at 16 kHz
CONTEXT = 64     # samples of the previous window the model is shown again


class EnergyGate:
    name = "energy"

    def __init__(self, threshold: float = 0.015):
        self.threshold = threshold
        self.last = 0.0

    def __call__(self, chunk: np.ndarray, rms: float, speaking: bool = False) -> bool:
        self.last = rms
        return rms > self.threshold


class SileroGate:
    name = "silero"

    def __init__(self, threshold: float = 0.5, release: float = 0.35, floor: float = 0.0):
        import onnxruntime
        import faster_whisper.vad as fw_vad
        path = os.path.join(os.path.dirname(fw_vad.__file__), "assets", "silero_vad_v6.onnx")
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        # Speech starts at `threshold`; once someone is speaking it has to fall
        # below `release` to count as quiet, so the dips between words do not
        self.threshold, self.release = threshold, release
        # Below this level nothing is speech, whatever the model says (0 = no floor)
        self.floor = floor
        self.last = 0.0
        self.reset()

    def reset(self):
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def probability(self, chunk: np.ndarray) -> float:
        """Highest speech probability among the model windows this chunk
        completes; the last value again when it completes none."""
        self._pending = np.concatenate([self._pending, np.asarray(chunk, dtype=np.float32).ravel()])
        best = None
        while len(self._pending) >= WINDOW:
            window, self._pending = self._pending[:WINDOW], self._pending[WINDOW:]
            frame = np.concatenate([self._context, window])[None, :]
            out, self._h, self._c = self.session.run(None, {"input": frame, "h": self._h, "c": self._c})
            self._context = window[-CONTEXT:]
            p = float(np.ravel(out)[0])
            best = p if best is None else max(best, p)
        if best is not None:
            self.last = best
        return self.last

    def __call__(self, chunk: np.ndarray, rms: float, speaking: bool = False) -> bool:
        p = self.probability(chunk)
        if rms < self.floor:
            return False
        return p >= (self.release if speaking else self.threshold)


def load(kind: str = "silero", energy_threshold: float = 0.015, threshold: float = 0.5):
    """The gate named in audio.vad; the energy gate when the model cannot load."""
    if str(kind).lower() == "silero":
        try:
            gate = SileroGate(threshold=threshold, release=max(0.05, threshold - 0.15))
            logger.info("Speech detector loaded (Silero VAD v6)")
            return gate
        except Exception as err:
            logger.error(f"Speech detector failed to load ({err}); falling back to loudness")
    return EnergyGate(energy_threshold)


class EchoGuard:
    """How loud the microphone must be, while a reply is playing, before it
    counts as the user talking over it.

    The speakers reach the microphone too, and with a fixed level a reply
    played a little louder interrupts itself. The app knows what it is playing,
    so the guard learns how much of it comes back: for every frame heard during
    playback it notes microphone level / playback level, and takes the 75th
    percentile of the last few seconds as the room's echo gain. Talking over a
    reply then has to be `margin` times louder than the echo expected right
    now, which follows the reply through its loud and quiet passages.

    `playback_level()` returns the level of what went to the speakers in the
    last moments (TTSEngine.recent_output_level). Without it the guard is the
    fixed level `base`, as before. Until it has heard half a second of
    playback it lets nothing interrupt, rather than guess.

    Simulated 2026-11-19 (replies echoed 120 ms late with a 150 ms room tail,
    20 runs each): at margin 2 and the 75th percentile no reply interrupted
    itself with the echo anywhere from 0.006 to 0.04, and a voice at 0.06 was
    heard over it in 18 to 20 of 20. Margin 1.5 or the median let replies
    interrupt themselves; the 90th percentile missed half the talk-overs."""

    def __init__(self, base: float = 0.015, margin: float = 2.0, playback_level=None,
                 percentile: float = 75.0, seconds: float = 6.0):
        self.base, self.margin, self.playback_level, self.percentile = base, margin, playback_level, percentile
        self._ratios = deque(maxlen=max(10, int(seconds / 0.05)))
        self._gain = None

    @property
    def gain(self):
        """Echo gain learned so far, or None before enough playback was heard."""
        return self._gain

    def level(self) -> float:
        if self.playback_level is None:
            return self.base
        if self._gain is None:
            return float("inf")
        return max(self.base, self.margin * self._gain * self.playback_level())

    def heard_playback(self, rms: float):
        if self.playback_level is None:
            return
        out = self.playback_level()
        if out > 1e-4:
            self._ratios.append(float(rms) / out)
            if len(self._ratios) >= 10:
                self._gain = float(np.percentile(self._ratios, self.percentile))
