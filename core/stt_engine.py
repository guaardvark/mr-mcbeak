"""
Faster-Whisper Speech-to-Text (STT) Engine.
Matches Guaardvark voice handling: 16 kHz float32, peak-normalize, VAD, anti-hallucination.
"""
import re
import time
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

# tiny.en / Whisper classic garbage on silence, TTS echo, and un-normalized clips
_HALLUCINATIONS = {
    "you",
    "thank you",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "thank you for watching this video",
    "bye",
    "the",
    "a",
    "uh",
    "um",
    "hmm",
    "you you",
    "thank you you",
    "subtitle",
    "subtitles by",
    "www youtube com",
}


def normalize_transcript(text: str) -> str:
    """Lowercase, strip punctuation — used for hallucination checks."""
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def is_whisper_hallucination(text: str) -> bool:
    """True for empty clips and known Whisper silence hallucinations like 'you'."""
    cleaned = normalize_transcript(text)
    return (not cleaned) or cleaned in _HALLUCINATIONS


def prepare_audio(audio_data: np.ndarray, target_peak: float = 0.8) -> np.ndarray:
    """
    Flatten to 1D float32 in [-1, 1] and peak-normalize.

    Guaardvark's browser path uses autoGainControl; without that, quiet BT/analog
    mics sit at RMS ~0.004 and tiny.en hallucinates 'you'.
    """
    if audio_data is None or audio_data.size == 0:
        return np.zeros(0, dtype=np.float32)

    audio = np.asarray(audio_data)
    if audio.ndim > 1:
        audio = audio[:, 0] if audio.shape[-1] <= 8 else audio.reshape(-1)

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / 32768.0

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-4:
        audio = audio * (target_peak / peak)
    return np.ascontiguousarray(audio, dtype=np.float32)


class STTEngine:
    """
    Manages cached Whisper models and performs fast in-memory transcription.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "auto",
        compute_type: str = "int8",
        beam_size: int = 5,
        vad_filter: bool = True
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.model: Optional[WhisperModel] = None

    def load_model(self):
        """Pre-warm and cache the Whisper model in memory."""
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError(
                "faster-whisper is not installed. Run ./setup.sh to create the Jarvis virtualenv."
            )

        if self.model is not None:
            return

        logger.info(f"Loading faster-whisper model '{self.model_size}' (device={self.device}, compute={self.compute_type})...")
        t0 = time.time()
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
        except Exception as e:
            logger.warning(f"Failed with compute_type={self.compute_type} ({e}), falling back to int8/cpu")
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8"
            )

        logger.info(f"Loaded faster-whisper '{self.model_size}' in {time.time() - t0:.2f}s")

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
        """
        Transcribe a 1D float32 numpy audio array.
        Returns (text, elapsed_seconds). Empty string if no real speech.
        """
        if self.model is None:
            self.load_model()

        t0 = time.time()
        audio = prepare_audio(audio_data)
        if audio.size < int(sample_rate * 0.2):
            return "", time.time() - t0

        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        logger.info(
            f"STT clip: samples={audio.size} dur={audio.size / sample_rate:.2f}s rms={rms:.4f} peak={peak:.4f}"
        )

        kwargs = {
            "beam_size": self.beam_size,
            "best_of": 3,
            "language": "en" if "en" in self.model_size else None,
            "vad_filter": self.vad_filter,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "without_timestamps": True,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
        if self.vad_filter:
            kwargs["vad_parameters"] = dict(min_silence_duration_ms=500)

        segments, _info = self.model.transcribe(audio, **kwargs)
        parts = []
        for seg in segments:
            no_speech = getattr(seg, "no_speech_prob", 0.0) or 0.0
            text = (seg.text or "").strip()
            if not text:
                continue
            if no_speech > 0.6:
                logger.debug(f"Dropping high no_speech_prob={no_speech:.2f} segment: {text!r}")
                continue
            if is_whisper_hallucination(text):
                logger.info(f"Dropping Whisper hallucination: {text!r}")
                continue
            parts.append(text)

        full_text = " ".join(parts).strip()
        if is_whisper_hallucination(full_text):
            logger.info(f"Dropping Whisper hallucination: {full_text!r}")
            full_text = ""

        return full_text, time.time() - t0
