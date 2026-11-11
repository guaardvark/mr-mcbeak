"""
Tone of voice from the audio itself, on this machine: emotion2vec+ large
(emotion2vec/emotion2vec_plus_large, FunASR Model Open Source License 1.1,
ONNX export Voldorn/emotion2vec-plus-large-onnx) on the CPU. It hears how
something was said, which a transcript loses.

Only a clear, non-neutral reading becomes a short note for the assistant
("sounds annoyed"); neutral or uncertain speech adds nothing. The note goes
to the assistant with the words, so it is off unless emotion.enabled is set.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

MODEL_REPO = "Voldorn/emotion2vec-plus-large-onnx"
MODEL_FILE = "emotion2vec_plus_large.onnx"
MODEL_REVISION = "cdcbc0964e2fa3092f79937287d4222056bc12fc"
LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]
# How each reading is put to the assistant
PHRASES = {"angry": "annoyed or angry", "disgusted": "displeased", "fearful": "worried or anxious",
           "happy": "upbeat", "sad": "down or tired", "surprised": "surprised"}


class ToneDetector:
    def __init__(self, model_path: Union[str, Path], threshold: float = 0.6, threads: int = 4):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold

    def scores(self, audio: np.ndarray) -> Dict[str, float]:
        """Probability per emotion for 16 kHz mono audio (at most the last 15 s)."""
        audio = np.asarray(audio, dtype=np.float32)[-16000 * 15:]
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        out = self.session.run(None, {self.input_name: audio[None, :]})
        probs = np.asarray(out[0]).reshape(-1)
        return {label: float(p) for label, p in zip(LABELS, probs)}

    def note(self, audio: np.ndarray) -> Optional[str]:
        """'sounds annoyed or angry', or None when the voice reads as neutral or unclear."""
        s = self.scores(audio)
        top = max(PHRASES, key=lambda k: s[k])
        logger.info("Tone: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(s.items(), key=lambda kv: -kv[1])[:3]))
        if s[top] >= self.threshold and s[top] > s["neutral"]:
            return f"sounds {PHRASES[top]}"
        return None


def model_path(models_dir: Union[str, Path]) -> Path:
    return Path(models_dir) / MODEL_FILE


def download_model(models_dir: Union[str, Path]) -> Path:
    from huggingface_hub import hf_hub_download
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION, local_dir=str(models_dir)))


def load(models_dir: Union[str, Path], threshold: float = 0.6) -> Optional[ToneDetector]:
    path = model_path(models_dir)
    if not path.exists():
        logger.warning(f"Tone model not found at {path}; tone of voice is off. Run ./setup.sh to fetch it.")
        return None
    detector = ToneDetector(path, threshold=threshold)
    logger.info(f"Tone model loaded ({path.name})")
    return detector
