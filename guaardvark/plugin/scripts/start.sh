#!/usr/bin/env bash
# Start the Mr. McBeak listener with its status endpoint: pid file, log, port check, health wait.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$PID_DIR" "$LOG_DIR" "$STATE_DIR"

if PID="$(running_pid)"; then
    echo "$PID" > "$PID_FILE"
    echo "mcbeak already running (pid $PID)"
    exit 0
fi
if command -v ss >/dev/null && ss -tln "sport = :$PORT" | grep -q ":$PORT"; then
    log "start refused: port $PORT already in use"
    echo "ERROR: port $PORT already in use" >&2
    exit 1
fi
# The microphone and speakers belong to the user's audio session.
if [[ -z "${XDG_RUNTIME_DIR:-}" && -d "/run/user/$(id -u)" ]]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi
# A copy started in a terminal holds the microphone; two would both answer.
PY="$MCBEAK_ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"
OTHER="$(cd "$MCBEAK_ROOT" && "$PY" -c 'from core.single_instance import other_copies; print(*other_copies())' 2>/dev/null || true)"
if [[ -n "$OTHER" ]]; then
    log "start refused: already listening outside this plugin (pid $OTHER)"
    echo "ERROR: Mr. McBeak is already listening in a terminal (pid $OTHER). Stop it with: mcbeak --stop" >&2
    exit 1
fi

# Guaardvark counts a sidecar as its own by a working directory inside the extension folder.
cd "$EXT_DIR"
# Members without a folder of their own work here instead of in that directory.
export MCBEAK_WORKSPACE="${MCBEAK_WORKSPACE:-$HOME}"
: > "$ERR_FILE"
chmod 600 "$ERR_FILE"
# The conversation the listener prints is discarded: nothing heard reaches a file from here.
nohup "$MCBEAK_ROOT/run.sh" --listen --status-port "$PORT" < /dev/null > /dev/null 2>> "$ERR_FILE" &
PID=$!
echo "$PID" > "$PID_FILE"
log "starting listener (pid $PID), status on 127.0.0.1:$PORT"

for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        log "listener up (pid $PID)"
        echo "mcbeak up on :$PORT (pid $PID)"
        exit 0
    fi
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
done
# Never leave a listener nobody can see holding the microphone.
if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    sleep 2
    kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null || true
fi
rm -f "$PID_FILE"
log "listener did not come up; its own messages are in $ERR_FILE"
echo "ERROR: mcbeak did not become healthy; see $ERR_FILE (run $MCBEAK_ROOT/setup.sh if it was never set up)" >&2
exit 1
