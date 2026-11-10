"""Read-only startup diagnostics; never submit assistant prompts."""
import os
from pathlib import Path
import shutil
import subprocess


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    if config.get("audio", {}).get("sample_rate", 16000) != 16000:
        raise ValueError("Set audio.sample_rate to 16000; resampling is not supported")
    assistants = config.get("assistants", {})
    if not assistants:
        raise ValueError("Configure at least one assistant")
    aliases = {a.get("target_id", key) for key, a in assistants.items()} | set(assistants)
    if config.get("system", {}).get("default_assistant", "claude") not in aliases:
        raise ValueError("system.default_assistant must name a configured assistant")
    for key, cfg in assistants.items():
        if cfg.get("permissions", "readonly") not in {"readonly", "full"}:
            raise ValueError(f"{key}: permissions must be readonly or full")
        if int(cfg.get("timeout_seconds", 120)) <= 0:
            raise ValueError(f"{key}: timeout_seconds must be positive")
    council = config.get("council", {})
    if council.get("reply_order", "completion") not in {"completion", "member"}:
        raise ValueError("council.reply_order must be completion or member")
    if any(m not in aliases for m in council.get("members", [])):
        raise ValueError("council.members contains an unknown assistant")


def doctor(config):
    failures = []

    def check(label, action):
        try:
            detail = action()
            print(f"OK {label}: {detail or 'available'}")
        except Exception as err:
            failures.append(label)
            print(f"FAIL {label}: {err}")

    check("configuration", lambda: validate_config(config))
    if "configuration" in failures:
        return 1
    from adapters.dispatcher import AdapterDispatcher
    from core.tts_engine import TTSEngine
    from core.stt_engine import FASTER_WHISPER_AVAILABLE
    import sounddevice as sd
    if not FASTER_WHISPER_AVAILABLE:
        failures.append("faster-whisper")
        print("FAIL faster-whisper: run ./setup.sh to install requirements")

    def audio():
        devices = sd.query_devices()
        sd.check_input_settings(samplerate=16000, channels=config.get("audio", {}).get("channels", 1))
        sd.check_output_settings()
        return f"{len(devices)} devices; default input supports 16 kHz"
    check("audio devices", audio)
    tts_cfg = config.get("tts", {})
    tts = TTSEngine(models_dir=tts_cfg.get("models_dir", "models/piper"),
                    extra_model_dirs=tts_cfg.get("extra_model_dirs", []))
    dispatcher = AdapterDispatcher(config["assistants"])
    for adapter in dict.fromkeys(dispatcher.adapters.values()):
        def workspace(a=adapter):
            path = a.workspace or os.getcwd()
            if not Path(path).is_dir():
                raise ValueError(f"workspace does not exist: {path}")
            return path
        check(f"{adapter.name} workspace", workspace)

        def binary(a=adapter):
            if not shutil.which(a.cli_binary):
                raise ValueError(f"install CLI {a.cli_binary} or set cli_cmd")
            result = subprocess.run([a.cli_binary, "--help"], capture_output=True, text=True, timeout=10)
            if result.returncode:
                raise ValueError("CLI --help failed; verify installation")
            if a.readonly:
                if not a.capabilities["readonly"]:
                    raise ValueError("read-only enforcement unavailable; adapter will refuse requests")
                flag = {"claude": "--permission-mode", "grok": "--deny", "cursor": "--mode",
                        "antigravity": "--mode"}[a.target_id]
                if flag not in result.stdout + result.stderr:
                    raise ValueError(f"CLI help does not advertise required permission flag {flag}")
            return a.cli_binary
        check(f"{adapter.name} CLI and permissions", binary)
        def voice(a=adapter):
            loaded = tts.get_or_load_voice(a.voice_file)
            if not 0 <= a.voice_speaker < loaded.config.num_speakers:
                raise ValueError(f"invalid voice_speaker {a.voice_speaker}")
            return a.voice_file
        check(f"{adapter.name} voice", voice)
    return int(bool(failures))
