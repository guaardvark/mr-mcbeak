"""
Multi-Voice Piper Text-to-Speech (TTS) Engine with Streaming Sentence Playback.
Supports multiple distinct persona voices, multi-speaker models, and instant barge-in killswitch.
"""
import os
import time
import queue
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import sounddevice as sd

from core.text_sanitizer import clean_text_for_tts, split_into_sentences

logger = logging.getLogger(__name__)

try:
    from piper import PiperVoice, SynthesisConfig
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    PiperVoice = None
    SynthesisConfig = None


class TTSEngine:
    """
    Manages cached Piper voice models and streams synthesized audio sentences to sounddevice.
    """

    def __init__(self, models_dir: str = "models/piper", playback_speed: float = 1.0,
                 extra_model_dirs: Optional[list] = None):
        self.playback_speed = playback_speed
        self.extra_model_dirs = [Path(os.path.expanduser(d)) for d in (extra_model_dirs or [])]
        p = Path(models_dir)
        if not p.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            p = project_root / models_dir
        self.models_dir = p
        self.loaded_voices: Dict[str, Any] = {}
        self.lock = threading.Lock()

        # Playback control & queue
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self._stop_requested = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None
        # Every PortAudio output call happens on the thread that holds _output_lock;
        # stop() only bumps _epoch. sounddevice's play()/wait()/stop() share one
        # module-global stream, and calling them from the speech, barge-in and
        # chime threads closed the same stream twice at once (SIGSEGV or deadlock).
        self._output_lock = threading.Lock()
        self._epoch_lock = threading.Lock()
        self._epoch = 0

    def get_or_load_voice(self, model_filename: str) -> Any:
        """Load and cache a Piper voice model with automatic fallback search."""
        if not PIPER_AVAILABLE:
            raise RuntimeError("piper-tts is not available. Run ./setup.sh to create the virtualenv.")

        with self.lock:
            if model_filename in self.loaded_voices:
                return self.loaded_voices[model_filename]

            # 1. Look in local models_dir
            model_path = self.models_dir / model_filename
            
            # 2. Fall back to any extra model dirs from config (tts.extra_model_dirs)
            if not model_path.exists():
                for d in self.extra_model_dirs:
                    if (d / model_filename).exists():
                        model_path = d / model_filename
                        break
                else:
                    raise FileNotFoundError(
                        f"Piper model not found: {model_filename} (searched {self.models_dir} and {self.extra_model_dirs})"
                    )

            logger.info(f"Loading Piper voice model: {model_path}...")
            voice = PiperVoice.load(str(model_path))
            self.loaded_voices[model_filename] = voice
            logger.info(f"Loaded voice: {model_filename}")
            return voice

    def stop(self):
        """Barge-in cancellation, safe from any thread.

        Playback that began before this call aborts its own stream within one
        block (~50 ms); this thread never touches PortAudio.
        """
        with self._epoch_lock:
            self._epoch += 1
        self._stop_requested.set()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.is_playing = False
        logger.debug("TTS playback stopped (barge-in interruption).")

    def _cancelled(self, epoch: int, context=None) -> bool:
        return epoch != self._epoch or bool(context and context.cancelled.is_set())

    def _stream_for(self, stream, sample_rate: int):
        """Return an open output stream at sample_rate, reusing `stream` when it matches."""
        if stream is not None:
            if stream.samplerate == sample_rate:
                return stream
            self._close(stream, drain=True)
        stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        return stream

    def _write(self, stream, audio: np.ndarray, epoch: int, context=None) -> bool:
        """Write audio in ~50 ms blocks, checking for cancellation between blocks.
        Returns True if every block was written."""
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        block = max(1, int(stream.samplerate * 0.05))
        for start in range(0, len(audio), block):
            if self._cancelled(epoch, context):
                return False
            stream.write(audio[start:start + block])
        return True

    @staticmethod
    def _close(stream, drain: bool):
        try:
            if drain:
                stream.stop()   # waits for queued audio to finish playing
            else:
                stream.abort()  # drops queued audio now
        finally:
            stream.close()

    def play_audio(self, audio: np.ndarray, sample_rate: int, context=None, mark_playing: bool = True) -> bool:
        """Play one buffer to completion unless stop() or the context cancels it.
        Returns True if it played to the end. mark_playing=False keeps the
        recorder listening, for cues too short and quiet to need echo handling."""
        epoch = self._epoch
        with self._output_lock:
            if self._cancelled(epoch, context):
                return False
            if mark_playing:
                self.is_playing = True
            stream, completed = None, False
            try:
                stream = self._stream_for(None, sample_rate)
                completed = self._write(stream, audio, epoch, context)
            finally:
                if stream is not None:
                    self._close(stream, drain=completed)
                if mark_playing:
                    self.is_playing = False
            return completed

    def speak(
        self,
        text: str,
        voice_file: str = "en_US-ryan-high.onnx",
        speaker_id: int = 0,
        speed: float = 1.0,
        context=None,
        block: bool = True,
        sentences=None,
    ):
        """
        Synthesize and speak text. If sentence_streaming is enabled, synthesizes
        and plays sentence-by-sentence for minimal first-audible-speech latency.
        """
        if sentences is None:
            # sentences, when given, are already cleaned and may still be
            # arriving (core/speech_stream.py)
            cleaned = clean_text_for_tts(text)
            if not cleaned:
                return
            sentences = split_into_sentences(cleaned) or [cleaned]

        epoch = self._epoch
        if self._cancelled(epoch, context):
            return
        self._stop_requested.clear()
        voice = self.get_or_load_voice(voice_file)
        # Piper's length_scale is the inverse of speed (0.9 = ~10% faster)
        rate = speed if speed and speed != 1.0 else (self.playback_speed or 1.0)
        syn_cfg = None
        if SynthesisConfig:
            syn_cfg = SynthesisConfig(speaker_id=speaker_id, length_scale=1.0 / max(rate, 0.25))

        with self._output_lock:
            if self._cancelled(epoch, context):
                return
            self.is_playing = True
            # Synthesis runs ahead of playback on its own thread: a sentence takes
            # ~0.4 s to synthesize while the device buffers ~35 ms, so synthesizing
            # in line left the speakers running dry at every sentence boundary.
            # Playback keeps one stream for the whole reply and is the only thread
            # that touches PortAudio.
            chunks: queue.Queue = queue.Queue(maxsize=8)
            done, stop = object(), threading.Event()

            def halted():
                return stop.is_set() or self._cancelled(epoch, context)

            def offer(item):
                while not halted():
                    try:
                        chunks.put(item, timeout=0.1)
                        return
                    except queue.Full:
                        pass

            def synthesize():
                try:
                    for sent in sentences:
                        if halted():
                            return
                        try:
                            for chunk in voice.synthesize(sent, syn_config=syn_cfg):
                                offer(chunk)
                                if halted():
                                    return
                        except Exception as e:
                            logger.error(f"Error synthesizing sentence '{sent[:30]}...': {e}")
                finally:
                    offer(done)

            producer = threading.Thread(target=synthesize, daemon=True, name="tts-synth")
            producer.start()
            stream, completed = None, True
            try:
                while True:
                    try:
                        chunk = chunks.get(timeout=0.1)
                    except queue.Empty:
                        if self._cancelled(epoch, context):
                            completed = False
                            break
                        continue
                    if chunk is done:
                        break
                    if self._cancelled(epoch, context):
                        completed = False
                        break
                    if context:
                        context.first_audio()
                    stream = self._stream_for(stream, chunk.sample_rate)
                    completed = self._write(stream, chunk.audio_float_array, epoch, context)
                    if not completed:
                        break
            except Exception as e:
                completed = False
                logger.error(f"Error playing TTS audio: {e}")
            finally:
                stop.set()
                if stream is not None:
                    try:
                        self._close(stream, drain=completed)
                    except Exception as e:
                        logger.error(f"Error closing TTS output stream: {e}")
                self.is_playing = False
                # espeak-ng (Piper's phonemizer) is not thread-safe: let this
                # synthesis finish before another reply can start one
                producer.join(timeout=5.0)

    def play_sound_cue(self, cue_type: str = "chime", context=None):
        """Generate and play a quick audio earcon (chime, grunt or thinking pulse).
        A cancelled context skips it, even if it was already waiting for the speaker."""
        try:
            sample_rate = 16000
            if cue_type == "chime":
                # Upward two-tone chime (440Hz -> 880Hz, 120ms)
                t1 = np.linspace(0, 0.06, int(sample_rate * 0.06), endpoint=False)
                tone1 = 0.2 * np.sin(2 * np.pi * 587.33 * t1)  # D5
                t2 = np.linspace(0, 0.08, int(sample_rate * 0.08), endpoint=False)
                tone2 = 0.25 * np.sin(2 * np.pi * 880.0 * t2)  # A5
                # Apply envelope
                tone1 *= np.linspace(1, 0.2, len(tone1))
                tone2 *= np.linspace(1, 0.05, len(tone2))
                audio = np.concatenate([tone1, tone2])
            elif cue_type == "thinking":
                # Soft low two-note pulse (G4 then E4), smooth envelopes so it never clicks
                notes = []
                for freq in (392.0, 329.63):
                    t = np.linspace(0, 0.07, int(sample_rate * 0.07), endpoint=False)
                    notes += [0.07 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t)),
                              np.zeros(int(sample_rate * 0.04))]
                audio = np.concatenate(notes)
            elif cue_type == "grunt":
                # Guaardvark low grunt (180Hz -> 120Hz)
                t = np.linspace(0, 0.15, int(sample_rate * 0.15), endpoint=False)
                freq = np.linspace(180, 120, len(t))
                audio = 0.3 * np.sin(2 * np.pi * freq * t)
                audio *= np.linspace(0.8, 0.01, len(audio))
            else:
                # Downward ready tone (660Hz -> 440Hz)
                t = np.linspace(0, 0.1, int(sample_rate * 0.1), endpoint=False)
                audio = 0.2 * np.sin(2 * np.pi * 523.25 * t)
                audio *= np.linspace(1, 0.1, len(audio))

            # The recorder keeps listening through a cue: marking it as playback
            # made the recorder deaf for the cue plus 0.4 s, which cut off words
            # spoken right after the chime. Whisper writes nothing for the tones.
            self.play_audio(audio.astype(np.float32), sample_rate, context=context, mark_playing=False)
        except Exception as e:
            logger.debug(f"Audio cue playback skipped: {e}")
