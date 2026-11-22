#!/usr/bin/env bash
# Stop the Mr. McBeak listener this plugin started, which closes the microphone. A copy run in a terminal is left alone.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mkdir -p "$LOG_DIR"

PID="$(running_pid || true)"
if [[ -z "$PID" ]]; then
    echo "mcbeak not running"
    rm -f "$PID_FILE"
    exit 0
fi
kill "$PID" 2>/dev/null || true
# A paused listener acts on the signal only once it runs again.
kill -CONT "$PID" 2>/dev/null || true
for _ in $(seq 1 20); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
done
if kill -0 "$PID" 2>/dev/null; then
    kill -9 "$PID" 2>/dev/null || true
    sleep 1
fi
if kill -0 "$PID" 2>/dev/null; then
    log "stop failed: pid $PID is still running"
    echo "ERROR: could not stop mcbeak (pid $PID)" >&2
    exit 1
fi
rm -f "$PID_FILE"
log "listener stopped (pid $PID)"
echo "mcbeak stopped (pid $PID)"
