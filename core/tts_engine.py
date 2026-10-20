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

    def __init__(self, models_dir: str = "models/piper"):
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

    def get_or_load_voice(self, model_filename: str) -> Any:
        """Load and cache a Piper voice model with automatic fallback search."""
        if not PIPER_AVAILABLE:
            raise RuntimeError("piper-tts is not available. Please run in the Jarvis virtualenv.")

        with self.lock:
            if model_filename in self.loaded_voices:
                return self.loaded_voices[model_filename]

            # 1. Look in local models_dir
            model_path = self.models_dir / model_filename
            
            # 2. Fallback to Guaardvark voice models if local copy missing
            if not model_path.exists():
                fallback_guaardvark = Path("~/guaardvark/backend/tools/voice/piper-models") / model_filename
                if fallback_guaardvark.exists():
                    model_path = fallback_guaardvark
                else:
                    raise FileNotFoundError(f"Piper model not found: {model_filename} (searched {self.models_dir})")

            logger.info(f"Loading Piper voice model: {model_path}...")
            voice = PiperVoice.load(str(model_path))
            self.loaded_voices[model_filename] = voice
            logger.info(f"Loaded voice: {model_filename}")
            return voice

    def stop(self):
        """Instant barge-in cancellation: stops audio output and flushes the queue."""
        self._stop_requested.set()
        sd.stop()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.is_playing = False
        logger.debug("TTS playback stopped (barge-in interruption).")

    def speak(
        self,
        text: str,
        voice_file: str = "en_US-ryan-high.onnx",
        speaker_id: int = 0,
        speed: float = 1.0,
        block: bool = True
    ):
        """
        Synthesize and speak text. If sentence_streaming is enabled, synthesizes
        and plays sentence-by-sentence for minimal first-audible-speech latency.
        """
        cleaned = clean_text_for_tts(text)
        if not cleaned:
            return

        sentences = split_into_sentences(cleaned)
        if not sentences:
            sentences = [cleaned]

        self._stop_requested.clear()
        voice = self.get_or_load_voice(voice_file)
        syn_cfg = SynthesisConfig(speaker_id=speaker_id) if (SynthesisConfig and speaker_id > 0) else None

        self.is_playing = True

        for sent in sentences:
            if self._stop_requested.is_set():
                break

            try:
                # Synthesize sentence
                for chunk in voice.synthesize(sent, syn_config=syn_cfg):
                    if self._stop_requested.is_set():
                        break

                    audio_arr = chunk.audio_float_array
                    sample_rate = chunk.sample_rate

                    # Play chunk
                    sd.play(audio_arr, samplerate=sample_rate)
                    sd.wait()

            except Exception as e:
                logger.error(f"Error synthesizing/playing sentence '{sent[:30]}...': {e}")

        self.is_playing = False

    def play_sound_cue(self, cue_type: str = "chime"):
        """Generate and play a quick audio earcon (chime or grunt) using synthetic tones."""
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

            sd.play(audio.astype(np.float32), samplerate=sample_rate)
            sd.wait()
        except Exception as e:
            logger.debug(f"Audio cue playback skipped: {e}")
