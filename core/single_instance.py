"""
One listening copy per user. A second copy fights the first over the microphone
and answers every question twice, and a paused one keeps the device open and
stalls the audio graph for every other program.

The guard is an flock on a file in the runtime directory. The kernel drops it
when the holder exits for any reason, crash or kill -9 included, so a stale lock
never needs cleaning up. Copies started before the guard existed hold no lock;
they are found by their command line instead.
"""
import fcntl
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Entry scripts of the listener, current and earlier names, and the flags that
# start it listening
SCRIPT_NAMES = ("mcbeak.py", "jarvis.py")
LISTEN_FLAGS = ("--listen", "--tmux", "--replace")


def lock_path() -> Path:
    base = (os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("XDG_STATE_HOME")
            or str(Path.home() / ".local" / "state"))
    return Path(base) / "mcbeak" / "listener.lock"


class InstanceLock:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else lock_path()
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Python opens it close-on-exec, so assistant CLIs started later never
        # inherit the lock and outlive us holding it
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        return True

    def release(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def holder(self) -> Optional[int]:
        """PID of the copy holding the lock, or None when nobody does."""
        if self._fd is not None:
            return os.getpid()
        if not self.acquire():
            try:
                pid = int(self.path.read_text().split()[0])
            except (OSError, ValueError, IndexError):
                return None
            return pid if _alive(pid) else None
        self.release()
        return None


def _stat(pid: int) -> List[str]:
    """Fields of /proc/<pid>/stat after the command name, so index 0 is the state."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return []
    return raw[raw.rfind(")") + 2:].split()


def _alive(pid: int) -> bool:
    fields = _stat(pid)
    return bool(fields) and fields[0] not in ("Z", "X")


def _cmdline(pid: int) -> List[str]:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")[:-1]
    except OSError:
        return []


def _is_listener(argv: List[str]) -> bool:
    return (bool(argv) and os.path.basename(argv[0]).startswith("python")
            and any(os.path.basename(a) in SCRIPT_NAMES for a in argv[1:2])
            and any(a in LISTEN_FLAGS for a in argv[2:]))


def other_copies(lock: Optional[InstanceLock] = None) -> List[int]:
    """Every other listening copy: the lock holder and any that predate the lock."""
    me = os.getpid()
    found = set()
    holder = (lock or InstanceLock()).holder()
    if holder:
        found.add(holder)
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and _is_listener(_cmdline(int(entry.name))):
            found.add(int(entry.name))
    return sorted(pid for pid in found if pid != me and _alive(pid))


def describe(pid: int) -> str:
    parts = [f"PID {pid}"]
    fields = _stat(pid)
    try:
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                    if line.startswith("btime"))
        started = datetime.fromtimestamp(boot + int(fields[19]) / os.sysconf("SC_CLK_TCK"))
        same_day = started.date() == datetime.now().date()
        parts.append("started " + started.strftime("%H:%M" if same_day else "%b %d %H:%M"))
    except (OSError, ValueError, IndexError, StopIteration):
        pass
    if fields and fields[0] in ("T", "t"):
        parts.append("paused")
    try:
        tty = os.readlink(f"/proc/{pid}/fd/0")
        if tty.startswith("/dev/pts/") or tty.startswith("/dev/tty"):
            parts.append("in terminal " + tty[5:])
    except OSError:
        pass
    return ", ".join(parts)


def stop_copies(pids: List[int], timeout: float = 8.0) -> List[int]:
    """Ask each copy to quit, kill what is left after timeout, return any still running."""
    def send(pid, sig):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass

    for pid in pids:
        send(pid, signal.SIGTERM)
        # a paused copy only acts on SIGTERM once it is running again
        send(pid, signal.SIGCONT)
    for sig, wait in ((None, timeout), (signal.SIGKILL, 3.0)):
        if sig:
            for pid in pids:
                if _alive(pid):
                    send(pid, sig)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.1)
    return [p for p in pids if _alive(p)]
