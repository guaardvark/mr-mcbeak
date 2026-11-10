#!/usr/bin/env bash
# Mr. McBeak launcher

# Resolve through the ~/.local/bin/mcbeak symlink to the real checkout
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"

# Priority: 1) local .venv, 2) Guaardvark venv, 3) ~/.venvs/mcbeak, 4) system python3
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "${GUAARDVARK_DIR:-$HOME/guaardvark}/backend/venv/bin/python" ]; then
    PYTHON_BIN="${GUAARDVARK_DIR:-$HOME/guaardvark}/backend/venv/bin/python"
elif [ -f "$HOME/.venvs/mcbeak/bin/python" ]; then
    PYTHON_BIN="$HOME/.venvs/mcbeak/bin/python"
else
    PYTHON_BIN="python3"
fi

# ONNX Runtime (Piper, Whisper VAD) telemetry stays off for every entry point
export ORT_DISABLE_TELEMETRY=1

exec "$PYTHON_BIN" "$SCRIPT_DIR/mcbeak.py" "$@"
