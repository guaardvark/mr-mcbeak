# Shared by start.sh and stop.sh.
SCRIPTS_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The folder Guaardvark links as extensions/mcbeak, and the checkout it sits in.
EXT_DIR="$(cd "$SCRIPTS_DIR/../.." && pwd -P)"
MCBEAK_ROOT="$(cd "$EXT_DIR/.." && pwd -P)"
PORT="${MCBEAK_STATUS_PORT:-8211}"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/mcbeak"
# Inside Guaardvark the pid and log sit with every other plugin's; alone, with the app's own state.
if [[ -n "${GUAARDVARK_ROOT:-}" ]]; then
    PID_DIR="$GUAARDVARK_ROOT/pids"; LOG_DIR="$GUAARDVARK_ROOT/logs"
else
    PID_DIR="$STATE_DIR/plugin"; LOG_DIR="$STATE_DIR/plugin"
fi
PID_FILE="$PID_DIR/mcbeak.pid"
# Guaardvark shows this log in the browser, so only these scripts write to it.
LOG_FILE="$LOG_DIR/mcbeak.log"
# The listener's own warnings and tracebacks can quote a reply; they stay beside its conversation log.
ERR_FILE="$STATE_DIR/plugin-listener.err"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG_FILE"
}

# A pid file can outlive its process and the number be reused; only a pid running this checkout's listener counts.
is_listener() {
    [[ "$1" =~ ^[0-9]+$ && -r "/proc/$1/cmdline" ]] || return 1
    local cmd exe
    exe="$(readlink "/proc/$1/exe" 2>/dev/null)" || return 1
    [[ "${exe##*/}" == python* ]] || return 1
    cmd="$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null)" || return 1
    [[ "$cmd" == *" $MCBEAK_ROOT/mcbeak.py --listen --status-port "* ]]
}

# The listener these scripts started: the pid file's, or, when the file is gone, the one found running.
running_pid() {
    local pid=""
    [[ -f "$PID_FILE" ]] && pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if is_listener "$pid"; then
        echo "$pid"
        return 0
    fi
    for pid in $(pgrep -f -- "--status-port" 2>/dev/null || true); do
        if is_listener "$pid"; then
            echo "$pid"
            return 0
        fi
    done
    return 1
}
