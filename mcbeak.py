#!/usr/bin/env python3
"""
Voice cockpit: hands-free spoken conversations with Claude Code, Grok,
Antigravity, Cursor and Guaardvark, one at a time or all together.
"""
import os
import sys
import time
import yaml
import signal
import logging
import argparse
import faulthandler
import re
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Telemetry off before anything loads ONNX Runtime or huggingface_hub or starts
# an assistant CLI (see core/privacy.py)
from core.privacy import disable_telemetry
disable_telemetry()

from core.text_sanitizer import clean_text_for_tts, extract_speakable_text
from core.wake_word_matcher import MultiTargetWakeWordMatcher
from core.stt_engine import STTEngine, is_whisper_hallucination
from core.tts_engine import TTSEngine
from core.audio_recorder import AudioRecorder
from core.state_manager import StateManager
from core.conversation import Conversation
from core.turn_context import TurnContext
from core.thinking_cue import start_thinking_cue
from core.council import build_prompt, run_council
from core.single_instance import InstanceLock, describe, other_copies, stop_copies
from adapters.dispatcher import AdapterDispatcher
from terminal.tmux_bridge import TmuxBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("cockpit")


def console_detail(verbose: bool):
    """The terminal shows the conversation; technical lines go to the log file
    (and to the terminal too with --verbose). Warnings and errors always show."""
    for handler in logging.getLogger().handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.INFO if verbose else logging.WARNING)


def _style(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if sys.stdout.isatty() else text


def show_you(name: str, prompt: str, how: str = ""):
    note = _style("2", f" ({how})") if how else ""
    print(f"\n{_style('1', 'You')} → {name}{note}: {prompt}", flush=True)


def show_reply(name: str, reply: str):
    print(f"{_style('1;36', name)}: {reply.strip()}", flush=True)


def show_note(text: str):
    print(_style("2", f"  {text}"), flush=True)


def load_config(config_path: str = "config.yaml") -> dict:
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found at: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_path() -> Path:
    """Where the conversation log is kept on this machine (never sent anywhere)."""
    state = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return state / "mcbeak" / "mcbeak.log"


def start_file_log():
    from logging.handlers import RotatingFileHandler
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(handler)


def app_name(config: dict) -> str:
    return str((config.get("system") or {}).get("name") or "Voice Cockpit")


def bare_name(phrase: str) -> str:
    """'hey grok' -> 'Grok'."""
    words = phrase.strip().split()
    while len(words) > 1 and words[0].lower() in ("hey", "ok", "okay", "hi", "hello", "yo"):
        words = words[1:]
    return " ".join(w.capitalize() for w in words)


def hint_words(config: dict) -> str:
    """Names Whisper should expect: the coordinator name and each assistant's first wake word."""
    stt_cfg = config.get("stt") or {}
    if stt_cfg.get("hint_words"):
        return str(stt_cfg["hint_words"])
    names = [app_name(config)] if config.get("coordinator_wake_words") else []
    for key, a in (config.get("assistants") or {}).items():
        names += [bare_name(w) for w in (a.get("wake_words") or [key])[:1]]
    return ", ".join(dict.fromkeys(n for n in names if n))


def whisper_lead_in(config: dict) -> str:
    """The names as a sentence Whisper reads before each clip, e.g. "Mr. McBeak,
    Claude, Grok, .... Hey Claude, hey Grok, Mr. McBeak, hey team." """
    stt_cfg = config.get("stt") or {}
    if str(stt_cfg.get("hint_style", "lead_in")).lower() != "lead_in":
        return ""
    names = [bare_name(w) for a in (config.get("assistants") or {}).values()
             for w in (a.get("wake_words") or [])[:1]]
    calls = [f"Hey {n}" if i == 0 else f"hey {n}" for i, n in enumerate(names[:2])]
    if config.get("coordinator_wake_words"):
        calls.append(app_name(config))
    calls.append("hey team")
    return f"{hint_words(config)}. {', '.join(calls)}."


def print_banner(config: dict, adapters: dict, default_assistant: str):
    """Who answers to what, in which voice and with which permissions."""
    rows = []
    for key, a in (config.get("assistants") or {}).items():
        adapter = adapters.get(a.get("target_id", key)) or adapters.get(key)
        if adapter is None:
            continue
        voice = str(a.get("voice", "")).replace("en_US-", "").replace(".onnx", "")
        if a.get("voice_speaker"):
            voice += f" #{a['voice_speaker']}"
        if adapter.readonly and not adapter.capabilities["readonly"]:
            mode = "off (cannot be read-only)"
        else:
            mode = "read-only" if adapter.readonly else "full"
        rows.append((f'"{bare_name((a.get("wake_words") or [key])[0])} ..."', adapter.name, voice, mode))
    coordinator = app_name(config) if config.get("coordinator_wake_words") else ""
    default_name = adapters[default_assistant].name if default_assistant in adapters else default_assistant
    width = max(len(r[0]) for r in rows) + 3 if rows else 20
    print(f"\n  {app_name(config)}  ·  listening\n")
    print(f"  {'Say':<{width}}{'Goes to':<16}{'Voice':<22}Permissions")
    for say, name, voice, mode in rows:
        print(f"  {say:<{width}}{name:<16}{voice:<22}{mode}")
    if coordinator:
        print(f'  {chr(34) + coordinator + " ..." + chr(34):<{width}}the last assistant you used (first: {default_name})')
    print(f'  {chr(34) + "Hey team ..." + chr(34):<{width}}everyone, in turn')
    print("\n  Keep talking for 30 s without a name. Talk over a reply to stop it.")
    print("  Pausing mid-sentence is fine: carry on within a few seconds and it joins up.")
    print('  Say "stop", "go to sleep", "new conversation" or "repeat that".')
    print(f"  Ctrl+C or Ctrl+Z quits. Log: {log_path()}\n")


def run_voice_daemon(config: dict, use_tmux: bool = False):
    """Main continuous voice processing loop."""
    assistants_cfg = config.get("assistants", {})
    audio_cfg = config.get("audio", {})
    stt_cfg = config.get("stt", {})
    tts_cfg = config.get("tts", {})
    system_cfg = config.get("system", {})

    from core.diagnostics import validate_config
    validate_config(config)
    start_file_log()
    logging.getLogger().setLevel(
        getattr(logging, str(system_cfg.get("log_level", "INFO")).upper(), logging.INFO)
    )

    # 1. Initialize Core Engines
    logger.info(f"Starting {app_name(config)}...")
    state_mgr = StateManager(
        active_timeout_seconds=system_cfg.get("active_timeout_seconds", 30.0),
        default_assistant=system_cfg.get("default_assistant", "claude")
    )
    council_cfg = config.get("council", {}) or {}
    matcher = MultiTargetWakeWordMatcher(
        config_assistants=assistants_cfg,
        coordinator_wake_words=config.get("coordinator_wake_words") or [],
        council_wake_words=council_cfg.get("wake_words", ["team", "hey team", "everyone", "hey everyone"]),
        coordinator_variants=config.get("coordinator_variants") or [],
    )
    display_names = {k: a.get("name", k.title()) for k, a in assistants_cfg.items()}
    for k, a in assistants_cfg.items():
        display_names[a.get("target_id", k)] = a.get("name", k.title())
    stt = STTEngine(
        model_size=stt_cfg.get("model_size", "tiny.en"),
        device=stt_cfg.get("device", "auto"),
        compute_type=stt_cfg.get("compute_type", "int8"),
        beam_size=stt_cfg.get("beam_size", 5),
        vad_filter=stt_cfg.get("vad_filter", True),
        hint_words=hint_words(config),
        language=stt_cfg.get("language", "en"),
        lead_in=whisper_lead_in(config),
    )
    stt.load_model()

    tts = TTSEngine(
        models_dir=tts_cfg.get("models_dir", "models/piper"),
        playback_speed=float(tts_cfg.get("playback_speed", 1.0)),
        extra_model_dirs=tts_cfg.get("extra_model_dirs") or [],
    )
    dispatcher = AdapterDispatcher(config_assistants=assistants_cfg)
    conversation = Conversation(state_mgr, tts, dispatcher)
    # tmux treats "." and ":" as target separators, so the session name is letters, digits and dashes
    tmux = TmuxBridge(session_name=re.sub(r"[^a-z0-9]+", "-", app_name(config).lower()).strip("-") or "voice-cockpit")
    council_members = [m for m in (council_cfg.get("members") or list(assistants_cfg)) if m in dispatcher.adapters]
    shared_turns = int(system_cfg.get("shared_transcript_turns", 6))
    continuation_s = float(audio_cfg.get("continuation_seconds", 4.0))

    def speak_reply(member: str, reply: str, voice_file: str, speaker_id: int, announce: bool = False, context=None):
        if context and context.cancelled.is_set():
            return
        show_reply(display_names.get(member, member), reply)
        spoken = extract_speakable_text(reply)
        if announce:
            spoken = f"{display_names.get(member, member)} says: {spoken}"
        logger.info(f"Speaking ({member}): {spoken[:120]!r}")
        conversation.speak(spoken, voice_file, speaker_id, context)

    if use_tmux:
        if tmux.is_available:
            tmux.create_cockpit_session(dispatcher.adapters)
            conversation.on_cancel = tmux.interrupt
            conversation.on_reset = tmux.reset
            print(f"\n💻 Tmux Cockpit session created. Attach with: tmux attach -t {tmux.session_name}\n")
        else:
            logger.warning("tmux not available, running in headless mode.")
            use_tmux = False

    # 2. Utterance Callback
    def on_utterance(audio_data):
        try:
            # Transcribe
            context = TurnContext(started_at=recorder.last_captured_at or time.monotonic())
            text, dur = stt.transcribe(audio_data)
            context.timing("transcription")
            if not text or len(text.strip()) < 2 or is_whisper_hallucination(text):
                return

            logger.info(f"🎙️  Heard ({dur:.2f}s): \"{text}\"")

            # Check wake word & active context
            active_target = state_mgr.active_assistant if state_mgr.is_in_active_window() else None
            match = matcher.match(text, active_target=active_target)

            if not match.detected:
                logger.debug("No wake-word or active session detected. Discarding.")
                show_note(f'heard, no name: "{text}"')
                return

            target = state_mgr.get_routing_target(match.target)
            shown_name = "Team" if target == "team" else display_names.get(target, target)
            logger.info(f"🎯 Matched Target: [{target.upper()}] (Similarity: {match.similarity:.2f})")

            prompt = match.remainder.strip()
            if prompt and (is_whisper_hallucination(prompt) or len(prompt) < 3):
                logger.info(f"Ignoring clipped remainder: {prompt!r}")
                prompt = ""

            if conversation.control(prompt, target):
                show_note(f'"{prompt}"')
                return

            if system_cfg.get("sound_cues", True) and not state_mgr.busy and not tts.is_playing:
                tts.play_sound_cue("chime")

            if not prompt:
                conversation.cancel()
                # Name-only wake: chime + listen. A spoken ack ate the follow-up
                # ("Hey <name>" → TTS → user's command discarded as self-echo).
                show_note(f"{shown_name} is listening")
                logger.info(f"{target} listening (30s follow-up window, no spoken ack)")
                return

            # Words that carry on the last question join it: said while the assistant
            # was still working, or within a few seconds of the last word and before
            # the reply had been spoken in full (a pause, not a new question)
            span = recorder.last_speech_span or (None, None)
            earlier, how = None, ""
            if not use_tmux:
                earlier, how = conversation.unanswered(target), "added while it was thinking"
                if not earlier:
                    earlier = conversation.continuation_of(target, span[0], continuation_s)
                    how = "carried on after a pause"
            if earlier:
                prompt = f"{earlier} {prompt}"
                logger.info(f"Joining with the previous question for {target} ({how})")
                show_you(shown_name, prompt, how)
            else:
                show_you(shown_name, prompt)

            def execute_turn(context):
                # Soft pulse while the assistant works (first after 1.5 s), so a long
                # wait does not sound like a dead microphone
                cue_on = system_cfg.get("sound_cues", True) and system_cfg.get("thinking_cue", True) and not use_tmux
                stop_cue = (start_thinking_cue(tts, context, float(system_cfg.get("thinking_cue_every", 3.0)))
                            if cue_on else (lambda: None))
                try:
                    if target == "team":
                        def transcript_for(member):
                            return state_mgr.recent_transcript(shared_turns, exclude_assistant=member, names=display_names)

                        def on_reply(member, reply, voice_file, speaker_id):
                            stop_cue()
                            if context.cancelled.is_set():
                                return
                            if not conversation.record_turn(member, prompt, reply, context):
                                return
                            speak_reply(member, reply, voice_file, speaker_id, announce=True, context=context)

                        run_council(
                            prompt, council_members, display_names, dispatcher.dispatch, transcript_for,
                            parallel=bool(council_cfg.get("parallel", True)), context=context,
                            reply_order=council_cfg.get("reply_order", "completion"), on_reply=on_reply,
                        )
                    elif use_tmux:
                        sent = tmux.send_prompt(target, prompt)
                        adapter = dispatcher.adapters[target]
                        ack = f"Command sent to {target}." if sent else f"Could not send command to {target}."
                        conversation.speak(ack, adapter.voice_file, adapter.voice_speaker, context)
                    else:
                        transcript = state_mgr.recent_transcript(shared_turns, exclude_assistant=target, names=display_names)
                        full_prompt = build_prompt(prompt, target, display_names.get(target, target), transcript,
                                                   list(display_names.values()))
                        response, voice_file, speaker_id = dispatcher.dispatch(target, full_prompt, context=context)
                        stop_cue()
                        if conversation.record_turn(target, prompt, response, context):
                            speak_reply(target, response, voice_file, speaker_id, context=context)
                except Exception:
                    logger.exception("Error executing turn %s", context.id)
                finally:
                    stop_cue()

            conversation.submit(context, execute_turn, target=target, prompt=prompt, spoken_until=span[1])

        except Exception as err:
            logger.error(f"Error handling voice utterance: {err}", exc_info=True)

    # 3. Start Continuous Recording
    turn_detector = None
    if audio_cfg.get("turn_detection", True):
        from core import turn_detector as td
        try:
            turn_detector = td.load(PROJECT_ROOT / "models" / "smart-turn")
        except Exception as err:
            logger.error(f"End-of-turn model failed to load ({err}); using silence timing only")
    recorder = AudioRecorder(
        sample_rate=audio_cfg.get("sample_rate", 16000),
        channels=audio_cfg.get("channels", 1),
        silence_threshold=audio_cfg.get("silence_threshold", 0.015),
        silence_duration_ms=audio_cfg.get("silence_duration_ms", 1800),
        min_speech_duration_ms=audio_cfg.get("min_speech_duration_ms", 400),
        max_utterance_seconds=audio_cfg.get("max_utterance_seconds", 30),
        max_queue_age_seconds=float(audio_cfg.get("max_queue_age_seconds", 8.0)),
        preroll_ms=int(audio_cfg.get("preroll_ms", 500)),
        on_utterance_ready=on_utterance,
        on_barge_in=conversation.cancel,
        tts_engine=tts,
        barge_in_enabled=audio_cfg.get("barge_in_enabled", True),
        turn_detector=turn_detector,
        turn_check_ms=int(audio_cfg.get("turn_check_ms", 500)),
        turn_threshold=float(audio_cfg.get("turn_threshold", 0.7)),
        max_silence_ms=int(audio_cfg.get("max_silence_ms", 3000)),
    )

    recorder.start()
    print_banner(config, dispatcher.adapters, system_cfg.get("default_assistant", "claude"))

    # Handle Ctrl+C cleanly
    def sig_handler(sig, frame):
        print(f"\nShutting down {app_name(config)}...")
        conversation.close()
        recorder.stop()
        tts.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    # Ctrl+Z would only pause it, still holding the microphone and stalling the
    # audio graph for other programs, so it quits instead
    signal.signal(signal.SIGTSTP, sig_handler)

    while True:
        time.sleep(0.5)


def claim_listener(config: dict, replace: bool) -> InstanceLock:
    """Make this the only listening copy, or explain why it cannot be and exit."""
    lock = InstanceLock()
    got = lock.acquire()
    others = other_copies(lock)
    if got and not others:
        return lock
    name = app_name(config)
    if not replace:
        lock.release()
        print(f"\n{name} is already running:", file=sys.stderr)
        for pid in others:
            print(f"  {describe(pid)}", file=sys.stderr)
        print("Only one copy can listen at a time, or both would answer every question.\n"
              "  mcbeak --replace   stop it and start this one\n"
              "  mcbeak --stop      just stop it\n", file=sys.stderr)
        sys.exit(1)
    for pid in others:
        print(f"Stopping the copy already running ({describe(pid)})...")
    left = stop_copies(others)
    if left:
        sys.exit(f"Could not stop PID {', '.join(map(str, left))}.")
    deadline = time.monotonic() + 3.0
    while not got and time.monotonic() < deadline:
        time.sleep(0.1)
        got = lock.acquire()
    if not got:
        sys.exit("Another copy started in the meantime; try again.")
    return lock


def stop_listener(config: dict) -> int:
    others = other_copies()
    if not others:
        print(f"{app_name(config)} is not running.")
        return 0
    for pid in others:
        print(f"Stopping {describe(pid)}...")
    left = stop_copies(others)
    if left:
        print(f"Could not stop PID {', '.join(map(str, left))}.", file=sys.stderr)
        return 1
    print("Stopped.")
    return 0


def test_voices(config: dict):
    """Test and preview the voices for all 5 assistants."""
    assistants = config.get("assistants", {})
    tts_cfg = config.get("tts", {})
    tts = TTSEngine(
        models_dir=tts_cfg.get("models_dir", "models/piper"),
        playback_speed=float(tts_cfg.get("playback_speed", 1.0)),
    )

    print("\n🔊 Testing Audio Voices for all assistants...\n")
    sample_phrases = {
        "claude": "Hello, I am Claude Code. Ready to analyze architecture and refactor code.",
        "grok": "Grok here! What are we building or debugging today?",
        "antigravity": "Antigravity online. Full agentic tooling and skills ready for deployment.",
        "cursor": "Cursor Agent active. Scanning workspace and prepared for code generation.",
        "guaardvark": "Guaardvark system operational. Standing by for instructions.",
    }

    for key, a in assistants.items():
        name = a.get("name", key)
        voice = a.get("voice", "en_US-ryan-high.onnx")
        speaker = a.get("voice_speaker", 0)
        phrase = sample_phrases.get(key, f"Hello from {name}.")

        print(f"🗣️  Speaking as [{name}] (Voice: {voice}, Speaker: {speaker})...")
        print(f"   \"{phrase}\"")
        tts.speak(phrase, voice_file=voice, speaker_id=speaker)
        time.sleep(0.5)

    print("\n✅ Voice test completed!\n")


def main():
    # A native crash (PortAudio, CTranslate2, onnxruntime) prints every thread's Python stack
    faulthandler.enable()
    parser = argparse.ArgumentParser(description="Voice cockpit for AI coding assistants")
    parser.add_argument("--doctor", action="store_true", help="Check configuration and local dependencies without submitting prompts")
    parser.add_argument("--listen", action="store_true", help="Listen continuously (also what plain `mcbeak` does)")
    parser.add_argument("--tmux", action="store_true", help="Launch 5-pane Tmux Cockpit and voice daemon")
    parser.add_argument("--replace", action="store_true", help="Stop the copy already listening, then start listening")
    parser.add_argument("--stop", action="store_true", help="Stop the copy that is listening")
    parser.add_argument("--test-voice", action="store_true", help="Preview voices for all assistants")
    parser.add_argument("--cli", nargs=2, metavar=("TARGET", "PROMPT"), help="Direct text execution (e.g. --cli claude 'what is 2+2')")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--verbose", action="store_true", help="Also show the technical log lines in the terminal")

    args = parser.parse_args()
    config = load_config(args.config)
    console_detail(args.verbose or str((config.get("system") or {}).get("console", "")).lower() == "verbose")

    if args.doctor:
        from core.diagnostics import doctor
        sys.exit(doctor(config))
    if args.stop:
        sys.exit(stop_listener(config))
    if args.test_voice:
        test_voices(config)
    elif args.cli:
        target, prompt = args.cli
        assistants_cfg = config.get("assistants", {})
        dispatcher = AdapterDispatcher(assistants_cfg)
        names = {k: a.get("name", k.title()) for k, a in assistants_cfg.items()}
        if target.lower() == "team":
            council_cfg = config.get("council", {}) or {}
            members = [m for m in (council_cfg.get("members") or list(assistants_cfg)) if m in dispatcher.adapters]
            for member, reply in run_council(prompt, members, names, dispatcher.dispatch, lambda m: "",
                                             parallel=bool(council_cfg.get("parallel", True)),
                                             reply_order=council_cfg.get("reply_order", "completion")):
                show_reply(names.get(member, member), reply)
        else:
            resp, voice, speaker = dispatcher.dispatch(target, build_prompt(prompt, target, names.get(target, target), "", []))
            show_reply(names.get(target, target), resp)
    else:
        # Plain `mcbeak` listens. Checked before any model loads or the
        # microphone opens; the lock is held until exit.
        lock = claim_listener(config, replace=args.replace)  # noqa: F841 (kept open)
        run_voice_daemon(config, use_tmux=args.tmux)


if __name__ == "__main__":
    main()
