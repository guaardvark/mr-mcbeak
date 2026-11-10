"""
Has the speaker finished their thought? Smart Turn v3 (pipecat-ai/smart-turn,
BSD-2-Clause) judges from how the last few seconds sound, a trailing "and..."
versus a finished question, instead of from how long the silence lasts. The
model is 8.7 MB and runs on the CPU in tens of milliseconds.

Input preparation follows the reference inference.py: the last 8 s of the turn,
zero-padded at the front, normalized to zero mean and unit variance, then
Whisper's 80-bin log-mel features. faster-whisper's extractor (already a
dependency) gives the same features as transformers' WhisperFeatureExtractor.
"""
import logging
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
MODEL_REPO = "pipecat-ai/smart-turn-v3"
MODEL_FILE = "smart-turn-v3.2-cpu.onnx"
MODEL_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"


class TurnDetector:
    def __init__(self, model_path: Union[str, Path]):
        import onnxruntime as ort
        from faster_whisper.feature_extractor import FeatureExtractor

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.features = FeatureExtractor(feature_size=80, sampling_rate=SAMPLE_RATE,
                                         chunk_length=WINDOW_SECONDS)

    def features_for(self, audio: np.ndarray) -> np.ndarray:
        n = SAMPLE_RATE * WINDOW_SECONDS
        audio = np.asarray(audio, dtype=np.float32)[-n:]
        audio = np.pad(audio, (n - len(audio), 0))
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        # padding=0: the window is already exactly 8 s, which gives 800 frames
        return self.features(audio.astype(np.float32), padding=0)

    def probability(self, audio: np.ndarray) -> float:
        """Probability (0-1) that the turn is complete."""
        mel = self.features_for(audio)
        out = self.session.run(None, {self.input_name: mel[None].astype(np.float32)})
        return float(np.asarray(out[0]).reshape(-1)[0])


def model_path(models_dir: Union[str, Path]) -> Path:
    return Path(models_dir) / MODEL_FILE


def download_model(models_dir: Union[str, Path]) -> Path:
    """Fetch the pinned model from Hugging Face once (telemetry is already off)."""
    from huggingface_hub import hf_hub_download
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION,
                                local_dir=str(models_dir)))


def load(models_dir: Union[str, Path], allow_download: bool = False):
    """The detector, or None (with a log line) when the model is missing."""
    path = model_path(models_dir)
    if not path.exists():
        if not allow_download:
            logger.warning(f"End-of-turn model not found at {path}; using silence timing only. "
                           "Run ./setup.sh to fetch it.")
            return None
        path = download_model(models_dir)
    detector = TurnDetector(path)
    logger.info(f"End-of-turn model loaded ({path.name})")
    return detector
