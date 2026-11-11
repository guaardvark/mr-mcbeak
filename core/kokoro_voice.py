"""
Kokoro v1.0 voices (hexgrad/Kokoro-82M, Apache-2.0) through kokoro-onnx on the
CPU, wrapped to look like a Piper voice so the speech engine treats both the
same: synthesize(text, syn_config) yields chunks with audio_float_array and
sample_rate. Selected in config.yaml as voice: "kokoro:<name>", e.g.
"kokoro:am_michael".

Measured 2026-11-10 against Piper (local-workspace-only/kokoro_vs_piper_*.txt):
first sentence in 0.31-0.52 s instead of 0.17 s, real-time factor ~0.1, and
64-65 of 67 words heard by Whisper for most male voices (Piper #430: 63).
am_fenrir and am_puck were much less clear (42 and 48). The int8 model was
slower on this CPU (3 s per sentence), so the full model is used.
"""
import threading
from pathlib import Path
from types import SimpleNamespace

PREFIX = "kokoro:"
MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"
RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"

_model = None
_model_lock = threading.Lock()


def is_kokoro(voice_file: str) -> bool:
    return str(voice_file).startswith(PREFIX)


def _load(models_dir: Path):
    global _model
    with _model_lock:
        if _model is None:
            from kokoro_onnx import Kokoro
            model, voices = models_dir / MODEL_FILE, models_dir / VOICES_FILE
            if not model.exists() or not voices.exists():
                raise FileNotFoundError(f"Kokoro model files missing in {models_dir}; run ./setup.sh")
            _model = Kokoro(str(model), str(voices))
        return _model


class KokoroVoice:
    def __init__(self, name: str, models_dir: Path):
        self.name = name
        self.model = _load(models_dir)
        if name not in self.model.get_voices():
            raise ValueError(f"Unknown Kokoro voice {name!r}")
        self.lang = "en-gb" if name.startswith("b") else "en-us"
        self.config = SimpleNamespace(num_speakers=1, sample_rate=24000)

    def synthesize(self, text, syn_config=None):
        length_scale = getattr(syn_config, "length_scale", None) or 1.0
        audio, sample_rate = self.model.create(text, voice=self.name, speed=1.0 / length_scale, lang=self.lang)
        yield SimpleNamespace(audio_float_array=audio.astype("float32"), sample_rate=sample_rate)
