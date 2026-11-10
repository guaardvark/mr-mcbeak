"""
Telemetry off, for the cockpit and for every assistant CLI it starts.

The only thing the cockpit sends off the machine is what the user says to the
assistant they named, through that assistant's own CLI. Usage metrics, crash
reports, trace uploads and update pings are switched off with each tool's own
documented opt-out. Values are forced, not defaulted, so an inherited
environment cannot turn them back on.
"""
import os
from typing import Dict, MutableMapping, Optional

NO_TELEMETRY_ENV: Dict[str, str] = {
    # ONNX Runtime (Piper voices, Whisper's VAD) queues Microsoft usage events
    "ORT_DISABLE_TELEMETRY": "1",
    # huggingface_hub usage pings when a Whisper model is first downloaded
    "HF_HUB_DISABLE_TELEMETRY": "1",
    # Cross-tool convention; Claude Code reads it
    "DO_NOT_TRACK": "1",
    # Claude Code and Grok: usage metrics and crash reports
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    # Claude Code: also update checks, /bug and surveys
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
    # Grok: product analytics, Mixpanel, and session trace upload (on by default)
    "GROK_TELEMETRY_ENABLED": "false",
    "GROK_TELEMETRY_MIXPANEL_ENABLED": "false",
    "GROK_TELEMETRY_TRACE_UPLOAD": "false",
    # Antigravity: crash reports (its usage telemetry is an account setting)
    "ANTIGRAVITY_SENTRY_SAMPLE_RATE": "0",
}


def disable_telemetry(env: Optional[MutableMapping[str, str]] = None) -> MutableMapping[str, str]:
    """Force every opt-out into `env` (default: this process's environment)."""
    target = os.environ if env is None else env
    target.update(NO_TELEMETRY_ENV)
    return target


def child_env() -> Dict[str, str]:
    """Environment for an assistant CLI subprocess, with every opt-out applied."""
    return disable_telemetry(dict(os.environ))
