#!/usr/bin/env python3
"""
Jarvis Multi-Agent Voice Cockpit ("Tony Stark Edition").
Hands-free voice chat interface for Claude, Grok, Antigravity, Cursor, and Guaardvark.
"""
import os
import sys
import time
import yaml
import signal
import logging
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.text_sanitizer import clean_text_for_tts, extract_speakable_text
from core.wake_word_matcher import MultiTargetWakeWordMatcher
from core.stt_engine import STTEngine, is_whisper_hallucination
from core.tts_engine import TTSEngine
from core.audio_recorder import AudioRecorder
from core.state_manager import StateManager
from core.council import build_prompt, run_council
from adapters.dispatcher import AdapterDispatcher
from terminal.tmux_bridge import TmuxBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("jarvis")


def load_config(config_path: str = "config.yaml") -> dict:
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found at: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_banner(assistants: dict):
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🎙️   JARVIS MULTI-AGENT VOICE COCKPIT                           ║
║                       (Tony Stark Edition)                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🟢 Status: LIVE & LISTENING                                                 ║
║  🗣️  Registered Assistants & Wake Words:                                     ║"""
    print(banner)
    for k, a in assistants.items():
        name = a.get("name", k)
        wake = a.get("wake_words", [k])[0]
        voice = a.get("voice", "en_US-ryan-high.onnx")
        print(f"║     • '{wake:<18}' ➔  {name:<16} (Voice: {voice[:15]})")
    print(f"║     • 'Hey Jarvis'        ➔  Default / last assistant                ║")
    print(f"║     • 'Hey Team'          ➔  Everyone answers (group conversation)    ║")
    print(f"║                                                                              ║")
    print(f"║  ⚡ Features:                                                                 ║")
    print(f"║     • Follow-up window: keep talking without repeating the wake-word       ║")
    print(f"║     • Each assistant remembers its conversation (CLI sessions resume)      ║")
    print(f"║     • Barge-in: speak over the assistant to stop it (headphones advised)   ║")
    print(f"║     • Read-only by default; set permissions: full per assistant to allow edits ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════╝\n")


def run_voice_daemon(config: dict, use_tmux: bool = False):
    """Main continuous voice processing loop."""
    assistants_cfg = config.get("assistants", {})
    audio_cfg = config.get("audio", {})
    stt_cfg = config.get("stt", {})
    tts_cfg = config.get("tts", {})
    system_cfg = config.get("system", {})

    print_banner(assistants_cfg)
    logging.getLogger().setLevel(
        getattr(logging, str(system_cfg.get("log_level", "INFO")).upper(), logging.INFO)
    )

    # 1. Initialize Core Engines
    logger.info("Initializing Jarvis AI Engines...")
    state_mgr = StateManager(
        active_timeout_seconds=system_cfg.get("active_timeout_seconds", 30.0),
        default_assistant=system_cfg.get("default_assistant", "claude")
    )
    council_cfg = config.get("council", {}) or {}
    matcher = MultiTargetWakeWordMatcher(
        config_assistants=assistants_cfg,
        coordinator_wake_words=config.get("coordinator_wake_words", ["jarvis", "hey jarvis"]),
        council_wake_words=council_cfg.get("wake_words", ["team", "hey team", "everyone", "hey everyone"]),
    )
    display_names = {k: a.get("name", k.title()) for k, a in assistants_cfg.items()}
    for k, a in assistants_cfg.items():
        display_names[a.get("target_id", k)] = a.get("name", k.title())
    stt = STTEngine(
        model_size=stt_cfg.get("model_size", "tiny.en"),
        device=stt_cfg.get("device", "auto"),
        compute_type=stt_cfg.get("compute_type", "int8"),
        beam_size=stt_cfg.get("beam_size", 5),
        vad_filter=stt_cfg.get("vad_filter", True)
    )
    stt.load_model()

    tts = TTSEngine(
        models_dir=tts_cfg.get("models_dir", "models/piper"),
        playback_speed=float(tts_cfg.get("playback_speed", 1.0)),
        extra_model_dirs=tts_cfg.get("extra_model_dirs") or [],
    )
    dispatcher = AdapterDispatcher(config_assistants=assistants_cfg)
    tmux = TmuxBridge()
    council_members = [m for m in (council_cfg.get("members") or list(assistants_cfg)) if m in dispatcher.adapters]
    shared_turns = int(system_cfg.get("shared_transcript_turns", 6))

    def speak_reply(member: str, reply: str, voice_file: str, speaker_id: int, announce: bool = False):
        print(f"\n[{display_names.get(member, member).upper()}]:\n{reply}\n")
        spoken = extract_speakable_text(reply)
        if announce:
            spoken = f"{display_names.get(member, member)} says: {spoken}"
        logger.info(f"Speaking ({member}): {spoken[:120]!r}")
        tts.speak(spoken, voice_file=voice_file, speaker_id=speaker_id)

    if use_tmux:
        if tmux.is_available:
            tmux.create_cockpit_session()
            print("\n💻 Tmux Cockpit session created. Attach with: tmux attach -t jarvis\n")
        else:
            logger.warning("tmux not available, running in headless mode.")

    # 2. Utterance Callback
    def on_utterance(audio_data):
        try:
            # Transcribe
            text, dur = stt.transcribe(audio_data)
            if not text or len(text.strip()) < 2 or is_whisper_hallucination(text):
                return

            logger.info(f"🎙️  Heard ({dur:.2f}s): \"{text}\"")

            # Check wake word & active context
            active_target = state_mgr.active_assistant if state_mgr.is_in_active_window() else None
            match = matcher.match(text, active_target=active_target)

            if not match.detected:
                logger.debug("No wake-word or active session detected. Discarding.")
                return

            target = state_mgr.get_routing_target(match.target)
            logger.info(f"🎯 Matched Target: [{target.upper()}] (Similarity: {match.similarity:.2f})")

            # Play subtle wake chime
            if system_cfg.get("sound_cues", True):
                tts.play_sound_cue("chime")

            prompt = match.remainder.strip()
            if prompt and (is_whisper_hallucination(prompt) or len(prompt) < 3):
                logger.info(f"Ignoring clipped remainder: {prompt!r}")
                prompt = ""

            if not prompt:
                # Name-only wake: chime + listen. A spoken ack ate the follow-up
                # ("Hey Jarvis" → TTS → user's command discarded as self-echo).
                print(f"\n[{target.upper()}]: Listening...\n")
                logger.info(f"{target} listening (30s follow-up window, no spoken ack)")
                return

            print(f"\n[You ➔ {target.upper()}]: \"{prompt}\"")

            if use_tmux and tmux.session_exists():
                tmux.send_prompt(target, prompt)
                ack = f"Command sent to {target}."
                tts.speak(ack, voice_file=dispatcher.adapters[target].voice_file, speaker_id=dispatcher.adapters[target].voice_speaker)
            elif target == "team":
                # Group conversation: everyone answers, in their own voice
                print(f"\n[TEAM]: asking {', '.join(display_names.get(m, m) for m in council_members)}...\n")

                def transcript_for(member):
                    return state_mgr.recent_transcript(shared_turns, exclude_assistant=member, names=display_names)

                def on_reply(member, reply, voice_file, speaker_id):
                    state_mgr.record_turn(member, prompt, reply)
                    speak_reply(member, reply, voice_file, speaker_id, announce=True)

                run_council(
                    prompt, council_members, display_names, dispatcher.dispatch, transcript_for,
                    parallel=bool(council_cfg.get("parallel", True)), on_reply=on_reply,
                )
            else:
                # Headless execution: the CLI session remembers its own turns; add
                # what the other assistants said so it can follow a group thread
                transcript = state_mgr.recent_transcript(shared_turns, exclude_assistant=target, names=display_names)
                full_prompt = build_prompt(prompt, target, display_names.get(target, target), transcript,
                                           list(display_names.values()))
                response, voice_file, speaker_id = dispatcher.dispatch(target, full_prompt)
                state_mgr.record_turn(target, prompt, response)
                speak_reply(target, response, voice_file, speaker_id)

        except Exception as err:
            logger.error(f"Error handling voice utterance: {err}", exc_info=True)

    # 3. Start Continuous Recording
    recorder = AudioRecorder(
        sample_rate=audio_cfg.get("sample_rate", 16000),
        channels=audio_cfg.get("channels", 1),
        silence_threshold=audio_cfg.get("silence_threshold", 0.015),
        silence_duration_ms=audio_cfg.get("silence_duration_ms", 800),
        min_speech_duration_ms=audio_cfg.get("min_speech_duration_ms", 400),
        max_utterance_seconds=audio_cfg.get("max_utterance_seconds", 30),
        max_queue_age_seconds=float(audio_cfg.get("max_queue_age_seconds", 8.0)),
        on_utterance_ready=on_utterance,
        tts_engine=tts if audio_cfg.get("barge_in_enabled", True) else None
    )

    recorder.start()

    # Handle Ctrl+C cleanly
    def sig_handler(sig, frame):
        print("\nShutting down Jarvis Voice Cockpit...")
        recorder.stop()
        tts.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    while True:
        time.sleep(0.5)


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
    parser = argparse.ArgumentParser(description="Jarvis Multi-Agent Voice Cockpit")
    parser.add_argument("--listen", action="store_true", help="Start continuous voice listener daemon")
    parser.add_argument("--tmux", action="store_true", help="Launch 5-pane Tmux Cockpit and voice daemon")
    parser.add_argument("--test-voice", action="store_true", help="Preview voices for all assistants")
    parser.add_argument("--cli", nargs=2, metavar=("TARGET", "PROMPT"), help="Direct text execution (e.g. --cli claude 'what is 2+2')")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    args = parser.parse_args()
    config = load_config(args.config)

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
                                             parallel=bool(council_cfg.get("parallel", True))):
                print(f"\n[{names.get(member, member).upper()}]:\n{reply}\n")
        else:
            resp, voice, speaker = dispatcher.dispatch(target, build_prompt(prompt, target, names.get(target, target), "", []))
            print(f"\n[{target.upper()}]:\n{resp}\n")
    elif args.tmux:
        run_voice_daemon(config, use_tmux=True)
    elif args.listen:
        run_voice_daemon(config, use_tmux=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
