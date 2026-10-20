#!/usr/bin/env bash
# Jarvis Multi-Agent Voice Cockpit Launcher

# Resolve through ~/.local/bin/jarvis symlink to the real jarvis_voice directory
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd "$(dirname "$SOURCE")" && pwd)"

# Priority: 1) local .venv, 2) Guaardvark venv, 3) ~/.venvs/jarvis, 4) system python3
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "${GUAARDVARK_DIR:-$HOME/guaardvark}/backend/venv/bin/python" ]; then
    PYTHON_BIN="${GUAARDVARK_DIR:-$HOME/guaardvark}/backend/venv/bin/python"
elif [ -f "$HOME/.venvs/jarvis/bin/python" ]; then
    PYTHON_BIN="$HOME/.venvs/jarvis/bin/python"
else
    PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/jarvis.py" "$@"
