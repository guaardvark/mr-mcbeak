"""
Continuous Audio Recorder & Voice Activity Detector (VAD).
Captures microphone input using sounddevice, computes energy envelope with hysteresis,
detects natural speech pauses for utterance segmentation, and handles barge-in talk-over.
"""
import time
import queue
import logging
import threading
from collections import deque
from typing import Callable, Optional, Any
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioRecorder:
    """
    Continuous audio recording stream with dynamic speech/silence segmentation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        silence_threshold: float = 0.015,
        silence_duration_ms: int = 800,
        min_speech_duration_ms: int = 400,
        max_utterance_seconds: int = 30,
        max_queue_age_seconds: float = 8.0,
        on_utterance_ready: Optional[Callable[[np.ndarray], None]] = None,
        tts_engine: Optional[Any] = None,
        on_barge_in: Optional[Callable] = None,
        barge_in_enabled: bool = True
    ):
        if sample_rate != 16000:
            raise ValueError("Audio capture requires sample_rate: 16000; resampling is not supported.")
        self.barge_in_enabled = barge_in_enabled
        self.last_captured_at = None
        self.on_barge_in = on_barge_in
        self._interrupt = threading.Event()
        self._interrupt_worker = None
        self._barge_active = False
        self._preroll = deque(maxlen=3)
        self.sample_rate = sample_rate
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_utterance_seconds = max_utterance_seconds
        self.max_queue_age_seconds = max_queue_age_seconds
        self.on_utterance_ready = on_utterance_ready
        self.tts_engine = tts_engine

        self._running = False
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

        # Offload STT/dispatch/TTS so the PortAudio callback never blocks
        self._utterance_queue: queue.Queue = queue.Queue(maxsize=3)
        self._worker: Optional[threading.Thread] = None

        # State tracking
        self._audio_buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._speech_start_time = 0.0
        self._last_speech_time = 0.0
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._tts_was_playing = False
        self._ignore_until = 0.0

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each chunk of incoming mic audio."""
        if status:
            logger.debug(f"Audio callback status: {status}")

        chunk = indata[:, 0].copy()  # Mono float32 array
        rms = np.sqrt(np.mean(chunk**2))
        now = time.monotonic()

        tts_playing = bool(self.tts_engine and self.tts_engine.is_playing)
        if self._tts_was_playing and not tts_playing and not self._barge_active:
            # Drop speaker-tail / BT sidetone so Whisper doesn't hallucinate "you"
            self._ignore_until = now + 0.4
            self._audio_buffer.clear()
            self._is_speaking = False
        self._tts_was_playing = tts_playing
        if now < self._ignore_until and not tts_playing:
            return

        # Talk-over / Barge-in: halt playback, then capture the interrupting speech
        if tts_playing and not self.barge_in_enabled:
            return
        if tts_playing and not self._barge_active:
            self._preroll.append(chunk)
            if rms > self.silence_threshold:
                self._consecutive_speech_frames += 1
                if self._consecutive_speech_frames >= 3:
                    logger.info("Barge-in detected! Stopping assistant speech.")
                    self._barge_active = True
                    self._audio_buffer = list(self._preroll)[:-1]
                    self._is_speaking = True
                    self._speech_start_time = now - 0.1
                    self._interrupt.set()
                    # Fall through so this interruption becomes the next utterance
                else:
                    return
            else:
                self._consecutive_speech_frames = 0
                return  # Discard self-echo while the assistant is still talking

        # Normal speech detection
        if rms > self.silence_threshold:
            self._consecutive_speech_frames += 1
            self._consecutive_silence_frames = 0
            self._last_speech_time = now

            if not self._is_speaking and self._consecutive_speech_frames >= 2:
                self._is_speaking = True
                self._speech_start_time = now
                self._audio_buffer.clear()
                logger.debug("Speech onset detected.")

        else:
            self._consecutive_silence_frames += 1
            self._consecutive_speech_frames = 0

        # Accumulate audio while speaking
        if self._is_speaking:
            self._audio_buffer.append(chunk)

            speech_duration = (now - self._speech_start_time) * 1000.0
            silence_duration = (now - self._last_speech_time) * 1000.0

            # Condition 1: Natural pause after minimum speech duration
            pause_detected = (
                speech_duration >= self.min_speech_duration_ms and
                silence_duration >= self.silence_duration_ms
            )

            # Condition 2: Max duration limit reached
            max_limit_reached = (speech_duration / 1000.0) >= self.max_utterance_seconds

            if pause_detected or max_limit_reached:
                logger.debug(f"Utterance complete (duration={speech_duration:.0f}ms, pause={silence_duration:.0f}ms).")
                self._emit_utterance()

    def _emit_utterance(self):
        """Assemble accumulated audio chunks and dispatch to callback."""
        if not self._audio_buffer:
            self._is_speaking = False
            return

        combined_audio = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()
        self._is_speaking = False
        self._barge_active = False
        self._preroll.clear()
        item = (time.monotonic(), combined_audio)
        try:
            self._utterance_queue.put_nowait(item)
        except queue.Full:
            try:
                self._utterance_queue.get_nowait()
            except queue.Empty:
                pass
            self._utterance_queue.put_nowait(item)

    def _worker_loop(self):
        """Process utterances off the audio callback thread."""
        while self._running or not self._utterance_queue.empty():
            try:
                captured_at, audio = self._utterance_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            age = time.monotonic() - captured_at
            if age > self.max_queue_age_seconds:
                # Heard while an assistant was still working; too old to act on now
                logger.info(f"Dropping stale utterance ({age:.1f}s old, {len(audio) / self.sample_rate:.1f}s of audio)")
                continue
            self.last_captured_at = captured_at
            if self.on_utterance_ready:
                try:
                    self.on_utterance_ready(audio)
                except Exception as e:
                    logger.error(f"Error in on_utterance_ready callback: {e}", exc_info=True)

    def _interrupt_loop(self):
        while self._running:
            if self._interrupt.wait(0.1):
                self._interrupt.clear()
                if self.on_barge_in:
                    self.on_barge_in()
                if self.tts_engine:
                    self.tts_engine.stop()

    def start(self):
        """Start capturing microphone input."""
        if self._running:
            return

        self._running = True
        self._interrupt_worker = threading.Thread(target=self._interrupt_loop, daemon=True)
        self._interrupt_worker.start()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="jarvis-utterance")
        self._worker.start()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            blocksize=int(self.sample_rate * 0.05),  # 50ms blocks
            callback=self._audio_callback
        )
        self._stream.start()
        logger.info("Audio recorder started (listening for wake-words)...")

    def stop(self):
        """Stop microphone capture."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug(f"Error closing audio stream: {e}")
            self._stream = None
        if self._interrupt_worker:
            self._interrupt_worker.join(timeout=1.0)
        if self._worker:
            self._worker.join(timeout=2.0)
            self._worker = None
        logger.info("Audio recorder stopped.")
