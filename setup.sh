#!/usr/bin/env bash
# ==============================================================================
# Jarvis Multi-Agent Voice Cockpit Setup Script
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
echo "🎙️  Setting up Jarvis Multi-Agent Voice Cockpit (Tony Stark Edition)..."
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

if [ ! -f "$MODELS_PIPER/en_US-ryan-high.onnx" ]; then
    if [ -f "${GUAARDVARK_PIPER_DIR}/en_US-ryan-high.onnx" ]; then
        echo "   • Copying Ryan voice from local Guaardvark tools..."
        cp ${GUAARDVARK_PIPER_DIR}/en_US-ryan-high.onnx* "$MODELS_PIPER/"
    else
        echo "   • Downloading Ryan high voice from HuggingFace..."
        wget -q --show-progress -O "$MODELS_PIPER/en_US-ryan-high.onnx" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx"
        wget -q --show-progress -O "$MODELS_PIPER/en_US-ryan-high.onnx.json" \
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json"
    fi
fi

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

# 3. Create Symlink in ~/.local/bin
mkdir -p "$HOME/.local/bin"
ln -sf "$SCRIPT_DIR/run.sh" "$HOME/.local/bin/jarvis"

# 4. Verify AI Assistant CLIs
echo "🤖 Verifying AI Assistant CLIs..."
for cli in claude grok agy agent guaardvark; do
    if command -v "$cli" >/dev/null 2>&1; then
        echo "   ✅ Found CLI: $cli ($(command -v $cli))"
    else
        echo "   ⚠️  CLI not found in PATH: $cli"
    fi
done

# 5. Run Verification Unit Tests
echo "🧪 Running unit tests..."
"$VENV_DIR/bin/python" -m unittest discover "$SCRIPT_DIR/tests"

echo "=========================================================================="
echo "✨ Jarvis Voice Cockpit setup complete!"
echo "🚀 Start with:"
echo "   jarvis --listen      # Continuous background voice listener"
echo "   jarvis --tmux        # Split-screen 5-pane interactive cockpit"
echo "   jarvis --test-voice  # Test audio voices for all 5 assistants"
echo "=========================================================================="
