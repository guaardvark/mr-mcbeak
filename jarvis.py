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
    print(f"║     • 'Hey Jarvis'        ➔  Broadcast / Default Coordinator        ║")
    print(f"║                                                                              ║")
    print(f"║  ⚡ Features:                                                                 ║")
    print(f"║     • 30s Active Follow-Up: Keep talking without repeating wake-words       ║")
    print(f"║     • Barge-In Interruption: Speak mid-sentence to immediately silence audio ║")
    print(f"║     • Dual-Mode: Headless background daemon or Tmux split-screen cockpit    ║")
    print(f"╚══════════════════════════════════════════════════════════════════════════════╝\n")


def run_voice_daemon(config: dict, use_tmux: bool = False):
    """Main continuous voice processing loop."""
    assistants_cfg = config.get("assistants", {})
    audio_cfg = config.get("audio", {})
    stt_cfg = config.get("stt", {})
    tts_cfg = config.get("tts", {})
    system_cfg = config.get("system", {})

    print_banner(assistants_cfg)

    # 1. Initialize Core Engines
    logger.info("Initializing Jarvis AI Engines...")
    state_mgr = StateManager(
        active_timeout_seconds=system_cfg.get("active_timeout_seconds", 30.0),
        default_assistant=system_cfg.get("default_assistant", "claude")
    )
    matcher = MultiTargetWakeWordMatcher(
        config_assistants=assistants_cfg,
        coordinator_wake_words=config.get("coordinator_wake_words", ["jarvis", "team"])
    )
    stt = STTEngine(
        model_size=stt_cfg.get("model_size", "tiny.en"),
        device=stt_cfg.get("device", "auto"),
        compute_type=stt_cfg.get("compute_type", "int8"),
        beam_size=stt_cfg.get("beam_size", 5),
        vad_filter=stt_cfg.get("vad_filter", True)
    )
    stt.load_model()

    tts = TTSEngine(models_dir=tts_cfg.get("models_dir", "~/guaardvark/backend/tools/voice/piper-models"))
    dispatcher = AdapterDispatcher(config_assistants=assistants_cfg)
    tmux = TmuxBridge()

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
            else:
                # Headless execution
                spoken_hint = (
                    "Answer in at most three short spoken sentences. "
                    "Do not narrate tools, subagents, or your process.\n\n"
                )
                response, voice_file, speaker_id = dispatcher.dispatch(target, spoken_hint + prompt)
                print(f"\n[{target.upper()}]:\n{response}\n")

                # State record
                state_mgr.record_turn(target, prompt, response)

                # Speak only the conversational reply, not the agent-team dump
                spoken = extract_speakable_text(response)
                logger.info(f"Speaking: {spoken[:120]!r}")
                tts.speak(spoken, voice_file=voice_file, speaker_id=speaker_id)

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
    tts = TTSEngine(models_dir=tts_cfg.get("models_dir", "~/guaardvark/backend/tools/voice/piper-models"))

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
        dispatcher = AdapterDispatcher(config.get("assistants", {}))
        resp, voice, speaker = dispatcher.dispatch(target, prompt)
        print(f"\n[{target.upper()}]:\n{resp}\n")
    elif args.tmux:
        run_voice_daemon(config, use_tmux=True)
    elif args.listen:
        run_voice_daemon(config, use_tmux=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
