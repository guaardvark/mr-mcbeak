"""Tmux cockpit with checked launches and stable pane identifiers."""
import os
import shlex
import shutil
import subprocess

from core.privacy import NO_TELEMETRY_ENV


class TmuxBridge:
    def __init__(self, session_name="voice-cockpit"):
        self.session_name = session_name
        self.tmux_bin = shutil.which("tmux")
        self.panes = {}
        self.launches = {}
        self.active_pane = None

    @property
    def is_available(self):
        return self.tmux_bin is not None

    def _run(self, *args):
        return subprocess.run([self.tmux_bin, *args], capture_output=True, text=True,
                              check=True, timeout=10)

    def session_exists(self):
        if not self.is_available:
            return False
        result = subprocess.run([self.tmux_bin, "has-session", "-t", self.session_name],
                                capture_output=True, timeout=10)
        return result.returncode == 0

    @staticmethod
    def launch_command(adapter):
        if adapter.readonly and not adapter.capabilities["readonly"]:
            raise ValueError(f"{adapter.name} cannot enforce read-only permissions")
        # Panes inherit the tmux server's environment, not the cockpit's, so the
        # telemetry opt-outs ride on the command itself
        cmd = ["env", *(f"{k}={v}" for k, v in NO_TELEMETRY_ENV.items()), adapter.cli_binary,
               *getattr(adapter, "launch_args", [])]
        if adapter.model:
            cmd += ["--model", adapter.model]
        if adapter.readonly:
            flags = {"claude": ["--permission-mode", "plan"],
                     "cursor": ["--mode", "ask"],
                     "antigravity": ["--mode", "plan"],
                     "grok": ["--deny", "Write", "--deny", "Edit", "--deny", "Bash"],
                     "codex": ["--sandbox", "read-only"]}
            cmd += flags[adapter.target_id]
        return shlex.join(cmd)

    def create_cockpit_session(self, adapters):
        if not self.is_available:
            raise RuntimeError("tmux is not installed")
        if self.session_exists():
            raise RuntimeError("Refusing existing tmux session: launch permissions cannot be verified. Use a new session name or close it yourself.")
        # Validate every launch before creating any panes. Unsupported read-only
        # adapters remain available as explicit errors through headless dispatch.
        launches = []
        for adapter in dict.fromkeys(adapters.values()):
            if adapter.readonly and not adapter.capabilities["readonly"]:
                continue
            launches.append((adapter, self.launch_command(adapter)))
        if not launches:
            raise RuntimeError("No adapters can be launched with configured permissions")
        for index, (adapter, command) in enumerate(launches):
            cwd = adapter.workspace or os.getcwd()
            if index == 0:
                result = self._run("new-session", "-d", "-s", self.session_name,
                                   "-n", "cockpit", "-P", "-F", "#{pane_id}", "-c", cwd, command)
            else:
                result = self._run("split-window", "-t", self.session_name + ":0", "-P",
                                   "-F", "#{pane_id}", "-c", cwd, command)
            pane = result.stdout.strip()
            self.launches[pane] = (cwd, command)
            for key, value in adapters.items():
                if value is adapter:
                    self.panes[key] = pane
            self._run("select-layout", "-t", self.session_name + ":0", "tiled")
        return True

    def send_prompt(self, target_id, prompt):
        pane = self.panes.get(target_id)
        if not pane:
            return False
        try:
            self.active_pane = pane
            self._run("send-keys", "-t", pane, "-l", "--", prompt)
            self._run("send-keys", "-t", pane, "Enter")
            return True
        except (subprocess.SubprocessError, OSError):
            return False

    def interrupt(self):
        pane, self.active_pane = self.active_pane, None
        if pane:
            try:
                self._run("send-keys", "-t", pane, "C-c")
            except (subprocess.SubprocessError, OSError):
                pass

    def reset(self, target):
        panes = set(self.panes.values()) if target == "team" else {self.panes.get(target)}
        for pane in panes - {None}:
            cwd, command = self.launches[pane]
            self._run("respawn-pane", "-k", "-t", pane, "-c", cwd, command)

    def capture_output(self, target_id, lines=15):
        pane = self.panes.get(target_id)
        if not pane:
            return ""
        return self._run("capture-pane", "-p", "-t", pane, "-S", str(-lines)).stdout.strip()
