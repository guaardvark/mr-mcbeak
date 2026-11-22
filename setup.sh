#!/usr/bin/env bash
# ==============================================================================
# Mr. McBeak setup
# Installs dependencies, sets up dedicated venv, ensures voice models,
# and verifies CLI assistants.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODELS_PIPER="$SCRIPT_DIR/models/piper"
MODELS_WHISPER="$SCRIPT_DIR/models/whisper"
# Reuse Guaardvark voice models if a checkout is present (override with GUAARDVARK_PIPER_DIR)
GUAARDVARK_PIPER_DIR="${GUAARDVARK_PIPER_DIR:-${GUAARDVARK_DIR:-$HOME/guaardvark}/backend/tools/voice/piper-models}"

echo "=========================================================================="
echo "🎙️  Setting up Mr. McBeak..."
echo "=========================================================================="

mkdir -p "$MODELS_PIPER" "$MODELS_WHISPER"

# 1. Create Dedicated Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating isolated Python virtual environment at $VENV_DIR..."
    PYTHON_CMD="python3"
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_CMD="python3.12"
    fi
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

echo "📦 Installing / verifying Python dependencies..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

# 2. Stage Piper Voice Models
echo "🔊 Checking Piper TTS voice models in $MODELS_PIPER..."

if [ ! -f "$MODELS_PIPER/en_US-libritts-high.onnx" ]; then
    if [ -f "${GUAARDVARK_PIPER_DIR}/en_US-libritts-high.onnx" ]; then
        echo "   • Copying LibriTTS voice from local Guaardvark tools..."
        cp ${GUAARDVARK_PIPER_DIR}/en_US-libritts-high.onnx* "$MODELS_PIPER/"
    else
        echo "   • Downloading LibriTTS multi-speaker voice from HuggingFace..."
        wget -q --show-progress -O "$MODELS_PIPER/en_US-libritts-high.onnx" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts/high/en_US-libritts-high.onnx"
        wget -q --show-progress -O "$MODELS_PIPER/en_US-libritts-high.onnx.json" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts/high/en_US-libritts-high.onnx.json"
    fi
fi

# 3. End-of-turn model (Smart Turn v3.2, pipecat-ai, BSD-2-Clause, 8.7 MB, runs on the CPU)
echo "⏱️  Checking the end-of-turn model..."
MCBEAK_DIR="$SCRIPT_DIR" "$VENV_DIR/bin/python" - <<'PY'
import os, sys
root = os.environ["MCBEAK_DIR"]
sys.path.insert(0, root)
from core.privacy import disable_telemetry
disable_telemetry()
from core import turn_detector
folder = os.path.join(root, "models", "smart-turn")
path = turn_detector.model_path(folder)
print(f"   • {'present' if path.exists() else 'downloaded'}: {path if path.exists() else turn_detector.download_model(folder)}")
PY

# 4. Kokoro voices (hexgrad/Kokoro-82M via kokoro-onnx, Apache-2.0, ~350 MB)
MODELS_KOKORO="$SCRIPT_DIR/models/kokoro"
mkdir -p "$MODELS_KOKORO"
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
    if [ ! -f "$MODELS_KOKORO/$f" ]; then
        echo "   • Downloading Kokoro $f..."
        wget -q --show-progress -O "$MODELS_KOKORO/$f" \
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f"
    fi
done

# 5. Tone-of-voice model (emotion2vec+ large, FunASR Model License 1.1, ~650 MB),
#    used when emotion.enabled is true in config.yaml or ~/.config/mcbeak/settings.yaml
MCBEAK_DIR="$SCRIPT_DIR" "$VENV_DIR/bin/python" - <<'PY'
import os, sys
root = os.environ["MCBEAK_DIR"]
sys.path.insert(0, root)
from core.privacy import disable_telemetry
disable_telemetry()
import yaml
from core import emotion, local_settings
config = local_settings.apply(yaml.safe_load(open(os.path.join(root, "config.yaml"))) or {})
if (config.get("emotion") or {}).get("enabled"):
    folder = os.path.join(root, "models", "emotion")
    path = emotion.model_path(folder)
    print(f"   • tone model {'present' if path.exists() else 'downloaded'}: {path if path.exists() else emotion.download_model(folder)}")
PY

# 6. Create Symlink in ~/.local/bin
mkdir -p "$HOME/.local/bin"
ln -sf "$SCRIPT_DIR/run.sh" "$HOME/.local/bin/mcbeak"

# 7. Verify AI Assistant CLIs
echo "🤖 Verifying AI Assistant CLIs..."
for cli in claude codex grok agy agent guaardvark ollama; do
    if command -v "$cli" >/dev/null 2>&1; then
        echo "   ✅ Found CLI: $cli ($(command -v $cli))"
    else
        echo "   ⚠️  CLI not found in PATH: $cli"
    fi
done

# 8. Run Verification Unit Tests
echo "🧪 Running unit tests..."
"$VENV_DIR/bin/python" -m unittest discover "$SCRIPT_DIR/tests"

echo "=========================================================================="
echo "✨ Mr. McBeak is set up."
echo "🚀 Start with:"
echo "   mcbeak               # Listen and talk"
echo "   mcbeak team          # Who is on the team; mcbeak team add NAME KIND to add someone"
echo "   mcbeak --test-voice  # Hear each team member's voice"
echo "=========================================================================="
