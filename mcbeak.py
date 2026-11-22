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
import threading
import logging
import argparse
import faulthandler
import re
from pathlib import Path
from typing import Optional

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
from core import approval
from core import local_settings
from core import team
from core.sessions import SessionStore
from core.speech_stream import SentenceFeed
from core.council import build_prompt, run_council
from core.single_instance import InstanceLock, describe, other_copies, stop_copies
from core.status_server import StatusServer, port_from
from core.diagnostics import validate_config
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
        return local_settings.apply(yaml.safe_load(f) or {})


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
    # Assistants that could be asked to join, so "ask Codex to join" is heard right
    names += list(config.get("known_names") or [])
    names += [g["name"] for g in (config.get("groups") or {}).values()]
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


def second_opinion(text: str, match, audio, name_stt, matcher, active_target=None):
    """When the main model heard no assistant name, let the name-check model
    (small.en, steadier on a short leading name) listen once more. Its version
    is taken only when it starts with a configured wake phrase exactly, so a
    near-homophone mid-sentence ("Mr. Gurr" heard as "Mr. Grok") does not wake
    anyone. Returns (text, match)."""
    if match.detected or name_stt is None:
        return text, match
    other, _ = name_stt.transcribe(audio)
    retry = matcher.match(other, active_target=active_target)
    words = " ".join(re.sub(r"[^a-z ]", " ", other.lower()).split())
    if retry.detected and retry.similarity >= 1.0 and words.startswith(retry.matched_phrase or "\0"):
        logger.info(f"Name check heard {other!r} (main model: {text!r})")
        return other, retry
    return text, match


class Roster:
    """The team as the listener uses it. Reloaded in place when the team file
    changes, so the objects handed out here (names, council, dispatcher) stay
    the same ones."""

    def __init__(self, config: dict, sessions=None, warm: bool = False):
        self.dispatcher = AdapterDispatcher({}, sessions=sessions)
        # The listener keeps each member ready between questions; one-off
        # commands do not
        self.warm = warm
        self.names: dict = {}
        self.council: list = []
        # Members sitting out team questions, with the reason (program missing,
        # not signed in, no answer); speaking to one directly brings it back
        self.benched: dict = {}
        # "group:<name>" -> member keys: part of the team asked together
        self.groups: dict = {}
        self.matcher = None
        self.load(config)

    def load(self, config: dict):
        """Returns (added keys, removed keys)."""
        self.config = config
        self.members = config.get("assistants") or {}
        council_cfg = config.get("council") or {}
        self.matcher = MultiTargetWakeWordMatcher(
            config_assistants=self.members,
            coordinator_wake_words=config.get("coordinator_wake_words") or [],
            council_wake_words=council_cfg.get("wake_words", ["team", "hey team", "everyone", "hey everyone"]),
            coordinator_variants=config.get("coordinator_variants") or [],
        )
        added, removed = self.dispatcher.update(self.members)
        names = {k: a.get("name", k.title()) for k, a in self.members.items()}
        for k, a in self.members.items():
            names.setdefault(a.get("target_id", k), a.get("name", k.title()))
        self.names.clear()
        self.names.update(names)
        self.council[:] = [m for m in (council_cfg.get("members") or list(self.members)) if m in self.dispatcher.adapters]
        self.groups.clear()
        for key, group in (config.get("groups") or {}).items():
            self.groups[key] = [m for m in group["members"] if m in self.dispatcher.adapters]
            self.names[key] = group["name"]
            self.matcher.register_target(key, group["wake_words"])
        self.dispatcher.groups = self.groups
        for key in removed:
            self.benched.pop(key, None)
        for key in added:
            self.benched.pop(key, None)
            reason = self.dispatcher.adapters[key].unavailable()
            if reason:
                self.benched[key] = reason
            elif self.warm:
                self.dispatcher.adapters[key].warm()
        return added, removed

    def answering(self, target: str = "team") -> list:
        """Who answers a question to the whole team or to a group: its members
        who are not sitting out."""
        return [m for m in self.groups.get(target, self.council) if m not in self.benched]

    def is_group(self, target) -> bool:
        return target == "team" or target in self.groups


_TEAM_WHO = re.compile(r"^(who is|who's|whos) (on|in) the team$|^who(?: is|'s|s)? here$|^who can i talk to$")
_TEAM_LEAVE = re.compile(r"^(?:remove|drop|take) (.+?) (?:from|off) the team$"
                         r"|^(?:ask|asks|asked|tell) (.+?) to leave(?: the team)?$")
_TEAM_JOIN = re.compile(r"^(?:add|invite|bring|ask|asks|asked|get|have) (.+?)"
                        r"(?: to join(?: the team| us| in)?| join(?: the team| us| in)?| to the team| onto the team| in)$")


def team_request(prompt: str):
    """A spoken request about the team itself: ("who", ""), ("join", name),
    ("leave", name), or None for anything else."""
    said = " ".join(re.sub(r"[^a-z0-9' ]", " ", (prompt or "").lower()).split())
    said = re.sub(r"^(please|can you|could you|would you) ", "", said)
    said = re.sub(r" please$", "", said)
    if not said or len(said.split()) > 9:
        return None
    if _TEAM_WHO.match(said):
        return ("who", "")
    m = _TEAM_LEAVE.match(said)
    if m:
        return ("leave", m.group(1) or m.group(2))
    m = _TEAM_JOIN.match(said)
    if m:
        return ("join", m.group(1))
    return None


def spoken_list(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else "nobody"


def print_banner(config: dict, roster: Roster, default_assistant: str):
    """Who answers to what, in which voice and with which permissions."""
    adapters = roster.dispatcher.adapters
    rows = []
    for key, a in roster.members.items():
        adapter = adapters.get(key)
        if adapter is None:
            continue
        voice = str(a.get("voice", "")).replace("en_US-", "").replace(".onnx", "")
        if a.get("voice_speaker"):
            voice += f" #{a['voice_speaker']}"
        if not getattr(adapter, "cli_binary", None) and adapter.target_id != "guaardvark":
            mode = "talks only (no tools)"
        elif adapter.readonly and not adapter.capabilities["readonly"]:
            mode = "off (cannot be read-only)"
        else:
            mode = "read-only" if adapter.readonly else "full"
        if key in roster.benched:
            mode = f"sitting out: {roster.benched[key]}"
        goes_to = adapter.name if adapter.name.lower().startswith(adapter.target_id) else f"{adapter.name} ({adapter.target_id})"
        rows.append((f'"{bare_name((a.get("wake_words") or [key])[0])} ..."', goes_to, voice, mode))
    coordinator = app_name(config) if config.get("coordinator_wake_words") else ""
    default_name = adapters[default_assistant].name if default_assistant in adapters else default_assistant
    width = max([len(r[0]) for r in rows] + [len(roster.names[k]) + 10 for k in roster.groups] + [17]) + 3
    wide = max([len(r[1]) for r in rows] + [13]) + 3
    print(f"\n  {app_name(config)}  ·  listening\n")
    print(f"  {'Say':<{width}}{'Goes to':<{wide}}{'Voice':<22}Permissions")
    for say, name, voice, mode in rows:
        print(f"  {say:<{width}}{name:<{wide}}{voice:<22}{mode}")
    if coordinator:
        print(f'  {chr(34) + coordinator + " ..." + chr(34):<{width}}the last assistant you used (first: {default_name})')
    print(f'  {chr(34) + "Hey team ..." + chr(34):<{width}}everyone, in turn')
    for key, members in roster.groups.items():
        print(f'  {chr(34) + "Hey " + roster.names[key] + " ..." + chr(34):<{width}}'
              f'{spoken_list(roster.names[m] for m in members)}')
    for note in config.get("team_notes") or []:
        print(f"  {note}")
    print("\n  Keep talking for 30 s without a name. Talk over a reply to stop it.")
    print("  Pausing mid-sentence is fine: carry on within a few seconds and it joins up.")
    print('  Say "stop", "go to sleep", "new conversation" or "repeat that".')
    asker = coordinator or "Hey team"
    print(f'  Team: "{asker}, who is on the team?", "{asker}, ask Grok to join", or `mcbeak team` in a terminal.')
    print(f"  Ctrl+C or Ctrl+Z quits. Log: {log_path()}\n")


def run_voice_daemon(config: dict, use_tmux: bool = False, config_path: str = "config.yaml",
                     status_port: Optional[int] = None):
    """Main continuous voice processing loop."""
    audio_cfg = config.get("audio", {})
    stt_cfg = config.get("stt", {})
    tts_cfg = config.get("tts", {})
    system_cfg = config.get("system", {})

    validate_config(config)
    start_file_log()
    logging.getLogger().setLevel(
        getattr(logging, str(system_cfg.get("log_level", "INFO")).upper(), logging.INFO)
    )

    # Answers from the first moment, so a host waiting on it sees "starting"
    # while the models load
    live = {"status": "starting", "roster": None, "recorder": None}
    if status_port is not None:
        def status_state():
            roster, recorder = live["roster"], live["recorder"]
            return {
                "status": live["status"],
                "listening": bool(recorder and recorder.listening),
                "team": [a.get("name", k) for k, a in roster.members.items()] if roster else [],
            }
        try:
            bound = StatusServer(status_port, status_state).start()
            logger.info(f"Status endpoint on http://127.0.0.1:{bound}/health")
        except OSError as err:
            logger.error(f"Status port {status_port} could not be opened ({err}); listening without it")

    # 1. Initialize Core Engines
    logger.info(f"Starting {app_name(config)}...")
    state_mgr = StateManager(
        active_timeout_seconds=system_cfg.get("active_timeout_seconds", 30.0),
        default_assistant=system_cfg.get("default_assistant", "claude")
    )
    council_cfg = config.get("council", {}) or {}
    sessions = None
    if system_cfg.get("remember_conversations", True) and not use_tmux:
        sessions = SessionStore(max_age_hours=float(system_cfg.get("remember_hours", 72)))
    roster = Roster(config, sessions, warm=not use_tmux)
    live["roster"] = roster
    display_names = roster.names
    # The kinds of assistant config.yaml describes, to recognise a spoken
    # "ask Grok to join" (with the same mishearings the wake words allow for)
    kind_matcher = MultiTargetWakeWordMatcher(config_assistants=load_config(config_path).get("assistants") or {})
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
    # A second, smaller model listens again only when the main one heard no
    # assistant name (large-v3-turbo sometimes drops a short leading "Claude,")
    name_stt = None
    if stt_cfg.get("name_check_model") and stt_cfg.get("name_check_model") != stt_cfg.get("model_size"):
        name_stt = STTEngine(
            model_size=stt_cfg["name_check_model"],
            device=stt_cfg.get("device", "auto"),
            compute_type=stt_cfg.get("compute_type", "int8"),
            beam_size=stt_cfg.get("beam_size", 5),
            vad_filter=stt_cfg.get("vad_filter", True),
            hint_words=hint_words(config),
            language=stt_cfg.get("language", "en"),
            lead_in=whisper_lead_in(config),
        )
        name_stt.load_model()

    tts = TTSEngine(
        models_dir=tts_cfg.get("models_dir", "models/piper"),
        playback_speed=float(tts_cfg.get("playback_speed", 1.0)),
        extra_model_dirs=tts_cfg.get("extra_model_dirs") or [],
    )
    dispatcher = roster.dispatcher
    # Load every voice now (Kokoro's model and phonemizer took ~0.7 s on first
    # use), so the first reply is not slower than the rest
    def load_voices(adapters):
        for adapter in adapters:
            try:
                voice = tts.get_or_load_voice(adapter.voice_file)
                if str(adapter.voice_file).startswith("kokoro:"):
                    list(voice.synthesize("Ready."))
            except Exception as err:
                logger.warning(f"Voice {adapter.voice_file} for {adapter.name} failed to load: {err}")
    load_voices(dispatcher.members())
    conversation = Conversation(state_mgr, tts, dispatcher)
    # tmux treats "." and ":" as target separators, so the session name is letters, digits and dashes
    tmux = TmuxBridge(session_name=re.sub(r"[^a-z0-9]+", "-", app_name(config).lower()).strip("-") or "voice-cockpit")
    # A question about the team itself waiting for a spoken yes: (asked at, action, name, kind)
    team_ask = {}
    shared_turns = int(system_cfg.get("shared_transcript_turns", 6))
    continuation_s = float(audio_cfg.get("continuation_seconds", 4.0))
    # Actions an assistant was held back from, waiting for a spoken yes
    held = approval.HeldActions()
    # Start speaking a reply's first sentence while the rest is still being written
    speak_early = bool(system_cfg.get("speak_early", True))
    # Tone of voice from the audio, read alongside transcription (core/emotion.py)
    emotion_cfg = config.get("emotion") or {}
    tone_detector, tone_pool = None, None
    if emotion_cfg.get("enabled"):
        from concurrent.futures import ThreadPoolExecutor
        from core import emotion
        try:
            tone_detector = emotion.load(PROJECT_ROOT / "models" / "emotion", float(emotion_cfg.get("threshold", 0.8)))
            tone_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tone")
        except Exception as err:
            logger.error(f"Tone model failed to load ({err}); tone of voice is off")

    def speak_reply(member: str, reply: str, voice_file: str, speaker_id: int, announce: bool = False, context=None):
        if context and context.cancelled.is_set():
            return
        show_reply(display_names.get(member, member), reply)
        spoken = extract_speakable_text(reply)
        if announce:
            spoken = f"{display_names.get(member, member)} says: {spoken}"
        logger.info(f"Speaking ({member}): {spoken!r}")
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

    def say(text: str, target=None):
        """Something the app itself says, in the voice of the assistant in use."""
        adapter = dispatcher.adapters.get(target) or dispatcher.adapters.get(state_mgr.default_assistant) \
            or dispatcher.members()[0]
        show_reply(app_name(config), text)
        logger.info(f"Speaking ({app_name(config)}): {text!r}")
        conversation.submit(TurnContext(), lambda ctx: conversation.speak(
            text, adapter.voice_file, adapter.voice_speaker, ctx))

    def team_by_voice(prompt: str, target) -> bool:
        """Handle "who is on the team", "ask Grok to join", "ask Grok to leave"
        and the yes or no that follows. True when the words were about the team."""
        if team_ask and time.monotonic() - team_ask["at"] <= approval.OPEN_SECONDS:
            said = approval.answer(prompt)
            ask = dict(team_ask)
            team_ask.clear()
            if said is not None:
                show_you(app_name(config), prompt)
                if said is False:
                    say("Okay, the team stays as it is.", target)
                    return True
                try:
                    raw = load_config(config_path)
                    if ask["action"] == "join":
                        team.add_member(raw, ask["name"], ask["kind"])
                    else:
                        team.remove_member(raw, ask["name"])
                    logger.info(f"Team changed by voice: {ask['action']} {ask['name']}")
                except team.TeamError as err:
                    say(str(err), target)
                return True   # the reload below announces who joined or left
        team_ask.clear()
        request = team_request(prompt)
        if request is None:
            return False
        action, who = request
        show_you(app_name(config), prompt)
        if action == "who":
            on = [display_names[k] for k in roster.members if k not in roster.benched]
            out = [display_names[k] for k in roster.members if k in roster.benched]
            text = f"On the team: {spoken_list(on)}."
            if out:
                text += f" Sitting out: {spoken_list(out)}."
            say(text, target)
            return True
        if action == "leave":
            found = roster.matcher.match(who)
            key = found.target if found.detected and found.target in roster.members else None
            if key is None:
                say(f"I did not catch who should leave. On the team: {spoken_list(display_names[k] for k in roster.members)}.", target)
                return True
            name = display_names[key]
            team_ask.update(at=time.monotonic(), action="leave", name=name, kind="")
            say(f"Take {name} off the team?", target)
            return True
        found = kind_matcher.match(who)
        kinds = team.templates(load_config(config_path))
        entry = next((e for k, (tk, e) in kinds.items() if found.detected and tk == found.target), None)
        if entry is None:
            say("By voice I can add " + spoken_list(e.get("name", k) for k, (_, e) in kinds.items())
                + ". For anyone else, use mcbeak team add in a terminal.", target)
            return True
        kind = team.kind_of(found.target, entry)
        name = entry.get("name") or found.target.title()
        if found.target in roster.members:
            say(f"{name} is already on the team.", target)
            return True
        team_ask.update(at=time.monotonic(), action="join", name=name, kind=kind)
        where = "What you say to it goes to its maker's servers. " if kind in team.CLOUD_KINDS else ""
        say(f"Add {name} to the team? {where}Say yes or no.", target)
        return True

    # 2. Utterance Callback
    def on_utterance(audio_data):
        try:
            # Transcribe
            context = TurnContext(started_at=recorder.last_captured_at or time.monotonic())
            tone_future = tone_pool.submit(tone_detector.note, audio_data) if tone_detector else None
            text, dur = stt.transcribe(audio_data)
            context.timing("transcription")
            if not text or len(text.strip()) < 2 or is_whisper_hallucination(text):
                return

            logger.info(f"🎙️  Heard ({dur:.2f}s): \"{text}\"")

            # Check wake word & active context
            active_target = state_mgr.active_assistant if state_mgr.is_in_active_window() else None
            match = roster.matcher.match(text, active_target=active_target)
            text, match = second_opinion(text, match, audio_data, name_stt, roster.matcher, active_target)

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

            if prompt and not use_tmux and team_by_voice(prompt, target):
                return

            if conversation.control(prompt, target):
                held.drop(target)
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

            # An answer to "Should it go ahead?" re-runs the held actions or drops
            # them; anything else is a new request and the question lapses
            approving = []
            if held.pending(target):
                said = approval.answer(prompt)
                actions = held.take(target)
                if said is False:
                    show_you(shown_name, prompt)
                    adapter = dispatcher.adapters.get(target)
                    show_reply(shown_name, "Okay, I won't.")
                    logger.info(f"Speaking ({target}): \"Okay, I won't.\"")
                    conversation.submit(TurnContext(), lambda ctx: conversation.speak(
                        "Okay, I won't.", adapter.voice_file, adapter.voice_speaker, ctx))
                    return
                if said is True and actions:
                    approving = actions
                    logger.info(f"Approved by voice ({target}): {actions}")

            # Words that carry on the last question join it: said while the assistant
            # was still working, or within a few seconds of the last word and before
            # the reply had been spoken in full (a pause, not a new question)
            span = recorder.last_speech_span or (None, None)
            earlier, how = None, ""
            if not use_tmux and not approving:
                earlier, how = conversation.unanswered(target), "added while it was thinking"
                if not earlier:
                    earlier = conversation.continuation_of(target, span[0], continuation_s)
                    how = "carried on after a pause"
            tone = ""
            if tone_future is not None:
                try:
                    tone = tone_future.result(timeout=2.0) or ""
                except Exception as err:
                    logger.warning(f"Tone check skipped: {err}")
            if earlier:
                prompt = f"{earlier} {prompt}"
                logger.info(f"Joining with the previous question for {target} ({how})")
                show_you(shown_name, prompt, ", ".join(x for x in (how, tone) if x))
            else:
                show_you(shown_name, prompt, tone)

            def execute_turn(context):
                # Soft pulse while the assistant works (first after 1.5 s), so a long
                # wait does not sound like a dead microphone
                cue_on = system_cfg.get("sound_cues", True) and system_cfg.get("thinking_cue", True) and not use_tmux
                stop_cue = (start_thinking_cue(tts, context, float(system_cfg.get("thinking_cue_every", 3.0)))
                            if cue_on else (lambda: None))
                try:
                    if roster.is_group(target):
                        def transcript_for(member):
                            return state_mgr.recent_transcript(shared_turns, exclude_assistant=member, names=display_names)

                        def on_reply(member, reply, voice_file, speaker_id):
                            stop_cue()
                            if context.cancelled.is_set():
                                return
                            failed = getattr(dispatcher.adapters.get(member), "last_error", None)
                            if failed:
                                # Said once; after that it sits out team questions until
                                # it answers when spoken to directly
                                roster.benched[member] = failed
                                logger.warning(f"{member} sits out team questions: {failed}")
                                speak_reply(member, f"{display_names.get(member, member)} could not answer and will "
                                            "sit out team questions until you speak to it directly.",
                                            voice_file, speaker_id, context=context)
                                return
                            if not conversation.record_turn(member, prompt, reply, context):
                                return
                            speak_reply(member, reply, voice_file, speaker_id, announce=True, context=context)

                        answering = roster.answering(target)
                        if not answering:
                            say("Nobody on the team can answer right now. Say who is on the team to hear why.")
                            return
                        run_council(
                            prompt, answering, display_names, dispatcher.dispatch, transcript_for,
                            parallel=bool(council_cfg.get("parallel", True)), context=context,
                            reply_order=council_cfg.get("reply_order", "completion"), on_reply=on_reply,
                            tone=tone, roles={m: dispatcher.adapters[m].role for m in answering},
                        )
                    elif use_tmux:
                        sent = tmux.send_prompt(target, prompt)
                        adapter = dispatcher.adapters[target]
                        ack = f"Command sent to {target}." if sent else f"Could not send command to {target}."
                        conversation.speak(ack, adapter.voice_file, adapter.voice_speaker, context)
                    else:
                        transcript = state_mgr.recent_transcript(shared_turns, exclude_assistant=target, names=display_names)
                        adapter = dispatcher.adapters.get(target)
                        full_prompt = build_prompt(prompt, target, display_names.get(target, target), transcript,
                                                   list(display_names.values()), tone=tone,
                                                   hint=not getattr(adapter, "spoken_system", False),
                                                   role=getattr(adapter, "role", ""))
                        if approving and adapter is not None:
                            adapter.approve_next = approving
                        feed, live = None, {}

                        def finish_speech():
                            # The log's only copy of a reply that was spoken as it arrived
                            live.pop("speech").join()
                            if feed.delivered:
                                cut = ", interrupted" if context.cancelled.is_set() else ""
                                logger.info(f"Spoke ({target}) as the reply arrived{cut}: {feed.delivered!r}")
                        if speak_early and getattr(adapter, "streams", False):
                            feed = SentenceFeed(max_sentences=3)

                            def on_text(piece):
                                if piece is None:
                                    feed.mute()
                                    return
                                if feed.add(piece) and "speech" not in live and not context.cancelled.is_set():
                                    stop_cue()
                                    logger.info(f"Speaking ({target}) as the reply arrives")
                                    live["speech"] = threading.Thread(
                                        target=conversation.speak_stream, name="reply-speech", daemon=True,
                                        args=(feed, adapter.voice_file, adapter.voice_speaker, context))
                                    live["speech"].start()
                        response, voice_file, speaker_id = dispatcher.dispatch(
                            target, full_prompt, context=context, on_text=on_text if feed else None)
                        stop_cue()
                        if feed:
                            feed.close()
                        if adapter is not None and not adapter.last_error and not context.cancelled.is_set():
                            roster.benched.pop(target, None)
                        waiting = list(getattr(adapter, "last_held", None) or [])
                        if waiting and not context.cancelled.is_set():
                            if feed:
                                feed.mute()
                            if "speech" in live:
                                finish_speech()
                            held.hold(target, waiting)
                            logger.info(f"Held for a spoken yes ({target}): {waiting}")
                            for action in waiting:
                                show_note(f"held for your yes: {action['tool']} {action['detail']}")
                            what = approval.describe(waiting, getattr(adapter, "confirm_first", []))
                            response = f"{display_names.get(target, target)} wants to {what}. Should it go ahead?"
                        if "speech" in live:
                            # Already being spoken: show it, keep the turn, wait for the voice
                            if conversation.record_turn(target, prompt, response, context):
                                show_reply(display_names.get(target, target), response)
                            finish_speech()
                        elif conversation.record_turn(target, prompt, response, context):
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
    from core import speech_gate
    silence_threshold = float(audio_cfg.get("silence_threshold", 0.015))
    recorder = AudioRecorder(
        speech_gate=speech_gate.load(audio_cfg.get("vad", "silero"), silence_threshold,
                                     float(audio_cfg.get("vad_threshold", 0.5))),
        echo_guard=speech_gate.EchoGuard(silence_threshold, float(audio_cfg.get("echo_margin", 2.0)),
                                         tts.recent_output_level),
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
    live["recorder"], live["status"] = recorder, "ok"
    print_banner(config, roster, system_cfg.get("default_assistant", "claude"))

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

    def team_stamp():
        try:
            return team.team_path().stat().st_mtime_ns
        except OSError:
            return None

    def reload_team():
        """The team file changed: bring the listener in line without a restart."""
        try:
            fresh = team.apply(load_config(config_path))
            validate_config(fresh)
        except (team.TeamError, ValueError, yaml.YAMLError, OSError) as err:
            logger.warning(f"Team file not loaded, the team stays as it was: {err}")
            return
        before = dict(display_names)
        groups_before = {k: list(v) for k, v in roster.groups.items()}
        added, removed = roster.load(fresh)
        state_mgr.default_assistant = (fresh.get("system") or {}).get("default_assistant", state_mgr.default_assistant)
        if state_mgr.active_assistant in removed or (
                str(state_mgr.active_assistant).startswith("group:") and state_mgr.active_assistant not in roster.groups):
            state_mgr.reset_to_passive()
        for key in removed:
            held.drop(key)
        for engine in (stt, name_stt):
            if engine is not None:
                engine.hint_words, engine.lead_in = hint_words(fresh) or None, whisper_lead_in(fresh) or None
        load_voices(dispatcher.adapters[k] for k in added)
        joined = [k for k in added if k not in before]
        left = [before[k] for k in removed]
        if not joined and not left:
            if groups_before != roster.groups:
                print_banner(fresh, roster, state_mgr.default_assistant)
            elif added:
                show_note("team settings updated: " + ", ".join(display_names[k] for k in added))
            return
        logger.info(f"Team reloaded: joined {joined}, left {left}")
        print_banner(fresh, roster, state_mgr.default_assistant)
        if state_mgr.busy or tts.is_playing:
            return
        for key in joined[:1]:
            adapter = dispatcher.adapters[key]
            text = f"{display_names[key]} joined the team."
            if key in roster.benched:
                text = f"{display_names[key]} is on the team but cannot answer yet: {roster.benched[key]}."
            show_reply(app_name(config), text)
            conversation.submit(TurnContext(), lambda ctx: conversation.speak(
                text, adapter.voice_file, adapter.voice_speaker, ctx))
        if left and not joined:
            say(f"{spoken_list(left)} left the team.")

    stamp = team_stamp()
    while True:
        time.sleep(0.5)
        now_stamp = team_stamp()
        if now_stamp != stamp and not use_tmux:
            stamp = now_stamp
            try:
                reload_team()
            except Exception:
                logger.exception("Reloading the team failed; the team stays as it was")


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


def mic_check(config: dict) -> int:
    """Measure the room: how quiet it is, how much of a reply comes back into
    the microphone, and how loud the user's voice arrives. Nothing is sent
    anywhere and no assistant is asked anything."""
    import numpy as np
    import sounddevice as sd
    from core import speech_gate
    if other_copies():
        print(f"{app_name(config)} is listening; stop it first (mcbeak --stop), or it will hear the test.", file=sys.stderr)
        return 1
    audio_cfg, tts_cfg = config.get("audio") or {}, config.get("tts") or {}
    base = float(audio_cfg.get("silence_threshold", 0.015))
    margin = float(audio_cfg.get("echo_margin", 2.0))
    tts = TTSEngine(models_dir=tts_cfg.get("models_dir", "models/piper"),
                    playback_speed=float(tts_cfg.get("playback_speed", 1.0)),
                    extra_model_dirs=tts_cfg.get("extra_model_dirs") or [])
    member = next(iter((config.get("assistants") or {}).values()))
    tts.get_or_load_voice(member.get("voice", "en_US-libritts-high.onnx"))
    gate = speech_gate.load("silero")
    guard = speech_gate.EchoGuard(base, margin, tts.recent_output_level)
    phase, rows = ["wait"], []

    def heard(indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        voiced = bool(gate(chunk, rms, False))
        needed = guard.level() if phase[0] == "reply" else base
        if phase[0] == "reply":
            guard.heard_playback(rms)
        rows.append((phase[0], rms, voiced, needed))

    def db(a, b):
        return 20 * np.log10(max(a, 1e-6) / max(b, 1e-6))

    def run_of(flags, n=3):
        streak = best = 0
        for f in flags:
            streak = streak + 1 if f else 0
            best = max(best, streak)
        return best >= n

    with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=800, callback=heard):
        time.sleep(0.5)
        print("\n1/3  Stay quiet for three seconds ...", flush=True)
        phase[0] = "room"
        time.sleep(3.0)
        print("2/3  A reply is playing; stay quiet ...", flush=True)
        phase[0] = "reply"
        tts.speak("This is how a reply sounds at the microphone. The speakers reach it too, "
                  "so I am measuring how much of my own voice comes back.",
                  voice_file=member.get("voice", "en_US-libritts-high.onnx"), speaker_id=int(member.get("voice_speaker", 0)))
        phase[0] = "wait"
        time.sleep(0.6)
        print(f'3/3  Now say, as you normally would: "{app_name(config)}, can you hear me?"  (five seconds)', flush=True)
        phase[0] = "voice"
        time.sleep(5.0)
        phase[0] = "wait"

    room = [r for r in rows if r[0] == "room"]
    reply = [r for r in rows if r[0] == "reply"]
    voice = [r for r in rows if r[0] == "voice" and r[2]]
    if not room or not reply:
        print("No audio arrived from the microphone.", file=sys.stderr)
        return 1
    floor = float(np.median([r[1] for r in room]))
    echo = float(np.percentile([r[1] for r in reply], 95))
    print(f"\n  Room         level {floor:.4f}; {sum(r[2] for r in room)} of {len(room)} frames sounded like speech")
    if max(r[1] for r in rows) < 0.001:
        print("               The microphone is delivering almost nothing. Check that it is the right one,\n"
              "               not muted, and that its input volume is not at zero.")
    print(f"  Reply echo   level {echo:.4f} at its loudest ({db(echo, floor):+.0f} dB over the room); "
          f"{sum(r[2] for r in reply)} of {len(reply)} frames sounded like speech")
    old = run_of([r[1] > base for r in reply])
    new = run_of([r[2] and r[1] > r[3] for r in reply])
    print(f"  Fixed level  ({base}): the reply {'WOULD interrupt itself' if old else 'would not interrupt itself'}")
    print(f"  Echo guard   (margin {margin}): the reply {'WOULD interrupt itself' if new else 'would not interrupt itself'}"
          + (f"; learned echo gain {guard.gain:.3f}" if guard.gain is not None else ""))
    need = float(np.median([r[3] for r in reply if np.isfinite(r[3])] or [base]))
    if not voice:
        print("  Your voice   nothing that sounded like speech was heard in the five seconds")
        print(f"               To talk over a reply at this volume a voice must reach about {need:.4f} at the microphone.")
    else:
        level = float(np.median([r[1] for r in voice]))
        print(f"  Your voice   level {level:.4f} ({db(level, floor):+.0f} dB over the room, {db(level, echo):+.0f} dB over the echo), "
              f"{len(voice) * 0.05:.1f} s of speech")
        print(f"               To talk over a reply at this volume it must reach about {need:.4f}: "
              + ("it does." if level > need else "it does NOT. Turn the speakers down, move the microphone "
                 "closer to you, or use headphones."))
    print()
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
        voice = a.get("voice", "en_US-libritts-high.onnx")
        speaker = a.get("voice_speaker", 0)
        phrase = sample_phrases.get(key, f"Hello from {name}.")

        print(f"🗣️  Speaking as [{name}] (Voice: {voice}, Speaker: {speaker})...")
        print(f"   \"{phrase}\"")
        tts.speak(phrase, voice_file=voice, speaker_id=speaker)
        time.sleep(0.5)

    print("\n✅ Voice test completed!\n")


def team_command(argv) -> int:
    """`mcbeak team`: see, add to and remove from the team. It only edits the
    team file; a copy that is listening picks the change up by itself."""
    parser = argparse.ArgumentParser(prog="mcbeak team", description="Who is on the team")
    parser.add_argument("--config", default="config.yaml", help=argparse.SUPPRESS)
    actions = parser.add_subparsers(dest="action")
    actions.add_parser("list", help="Show the team (also what plain `mcbeak team` does)")
    add = actions.add_parser("add", help="Add a member, e.g. `mcbeak team add Grok` or `mcbeak team add Ada claude`")
    add.add_argument("name", help="What you will call it out loud")
    add.add_argument("kind", nargs="?", help=f"{', '.join(team.KINDS)} (not needed when the name is the kind)")
    add.add_argument("--model", help="Model to use (required for ollama; otherwise the program's own default)")
    add.add_argument("--folder", help="Folder it works in (default: where mcbeak is started)")
    add.add_argument("--permissions", choices=["readonly", "full"], help="readonly (default) or full")
    add.add_argument("--voice", help='e.g. "kokoro:af_bella" (default: a voice nobody else has)')
    add.add_argument("--role", help="A standing brief, added to everything it is asked")
    add.add_argument("--host", help="ollama: where Ollama runs (default http://127.0.0.1:11434)")
    remove = actions.add_parser("remove", help="Take a member off the team")
    remove.add_argument("name")
    group = actions.add_parser("group", help='Part of the team under one name, e.g. `mcbeak team group reviewers Ada Grok` '
                                             'for "Hey reviewers ..."; no members deletes the group')
    group.add_argument("name", help="What you will call the group out loud")
    group.add_argument("members", nargs="*", help="Member names")
    args = parser.parse_args(argv)
    raw = load_config(args.config)
    path = team.team_path()
    try:
        if args.action == "add":
            kind = team.KIND_ALIASES.get(str(args.kind or team.slug(args.name)).lower(),
                                         str(args.kind or team.slug(args.name)).lower())
            if kind == "ollama":
                from adapters.ollama_adapter import OllamaAdapter, DEFAULT_HOST, installed_models
                host = args.host or DEFAULT_HOST
                if not args.model:
                    try:
                        have = ", ".join(installed_models(host)) or "none yet (ollama pull MODEL)"
                    except Exception:
                        have = f"unknown, Ollama is not answering at {host}"
                    raise team.TeamError(f"Say which model {args.name} is, with --model. Installed: {have}")
                reason = OllamaAdapter({"name": args.name, "model": args.model, "host": host}).unavailable()
                if reason:
                    raise team.TeamError(f"{args.name} cannot join: {reason}.")
            entry = team.add_member(raw, args.name, args.kind, path=path, model=args.model, folder=args.folder,
                                    permissions=args.permissions, voice=args.voice, role=args.role, host=args.host)
            said = bare_name((entry.get("wake_words") or [args.name])[0])
            print(f'{entry["name"]} is on the team. Say "Hey {said}, ..." (voice {entry["voice"]}, '
                  f'permissions {entry.get("permissions", "readonly")}).')
            if entry["kind"] in team.CLOUD_KINDS:
                print(f"What you say to {entry['name']} is sent by its own program to its maker; nothing else is.")
            else:
                print(f"{entry['name']} runs on your own machines; nothing you say to it leaves them.")
            print(f"Team file: {path}\nA copy that is listening picks this up by itself.")
            return 0
        if args.action == "remove":
            member = team.remove_member(raw, args.name, path=path)
            print(f"{member['name']} is off the team. Team file: {path}")
            return 0
        if args.action == "group":
            made = team.set_group(raw, args.name, args.members, path=path)
            if made is None:
                print(f"The group {args.name} is gone. Team file: {path}")
            else:
                names = team.apply(load_config(args.config), path=path)["assistants"]
                print(f'Say "Hey {made["name"]}, ..." to ask '
                      f'{spoken_list(names[m].get("name", m) for m in made["members"] if m in names)}. Team file: {path}')
            return 0
        config = team.apply(raw, path=path)
    except team.TeamError as err:
        print(err, file=sys.stderr)
        return 1
    rows = [("Say", "Name", "Kind", "Model", "Folder", "Voice", "Permissions")]
    for key, a in config["assistants"].items():
        rows.append((f'"Hey {bare_name((a.get("wake_words") or [key])[0])}"', a.get("name", key), a.get("kind", ""),
                     str(a.get("model") or "default"),
                     "-" if a.get("kind") in ("ollama", "guaardvark") else str(a.get("folder") or a.get("workspace") or "where started"),
                     str(a.get("voice", "")).replace("en_US-", "").replace(".onnx", "")
                     + (f" #{a['voice_speaker']}" if a.get("voice_speaker") else ""),
                     "talks only" if a.get("kind") == "ollama" else a.get("permissions", "readonly")))
    widths = [max(len(r[i]) for r in rows) + 2 for i in range(len(rows[0]))]
    print()
    for row in rows:
        print("  " + "".join(f"{cell:<{w}}" for cell, w in zip(row, widths)).rstrip())
    for note in config.get("team_notes") or []:
        print(f"  {note}")
    for key, grp in (config.get("groups") or {}).items():
        print(f'  "Hey {grp["name"]}" asks ' + spoken_list(config["assistants"][m].get("name", m) for m in grp["members"]))
    if config.get("team_file"):
        print(f"\n  Team file: {config['team_file']}")
    else:
        print(f"\n  This is the starting team: every assistant in config.yaml that is installed here.\n"
              f"  Your first `mcbeak team add` or `remove` writes it to {path}; after that the file is the team.")
    print("  mcbeak team add NAME [KIND] [--model M] [--folder DIR] [--permissions full] [--role TEXT]\n"
          "  mcbeak team remove NAME\n"
          "  mcbeak team group GROUP NAME NAME ...     (\"Hey GROUP ...\" asks just them)\n")
    return 0


def main():
    # A native crash (PortAudio, CTranslate2, onnxruntime) prints every thread's Python stack
    faulthandler.enable()
    parser = argparse.ArgumentParser(description="Voice cockpit for AI coding assistants",
                                     epilog="`mcbeak team` shows the team; `mcbeak team add NAME [KIND]` and "
                                            "`mcbeak team remove NAME` change it.")
    parser.add_argument("--doctor", action="store_true", help="Check configuration and local dependencies without submitting prompts")
    parser.add_argument("--listen", action="store_true", help="Listen continuously (also what plain `mcbeak` does)")
    parser.add_argument("--tmux", action="store_true", help="Launch 5-pane Tmux Cockpit and voice daemon")
    parser.add_argument("--replace", action="store_true", help="Stop the copy already listening, then start listening")
    parser.add_argument("--stop", action="store_true", help="Stop the copy that is listening")
    parser.add_argument("--test-voice", action="store_true", help="Preview voices for all assistants")
    parser.add_argument("--mic-check", action="store_true",
                        help="Measure room noise, the echo of a reply at the microphone, and your voice")
    parser.add_argument("--cli", nargs=2, metavar=("TARGET", "PROMPT"), help="Direct text execution (e.g. --cli claude 'what is 2+2')")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--verbose", action="store_true", help="Also show the technical log lines in the terminal")
    parser.add_argument("--status-port", type=int, metavar="N",
                        help="While listening, answer GET /health on 127.0.0.1:N (also $MCBEAK_STATUS_PORT; off by default)")

    if len(sys.argv) > 1 and sys.argv[1] == "team":
        sys.exit(team_command(sys.argv[2:]))
    args = parser.parse_args()
    try:
        config = team.apply(load_config(args.config))
    except team.TeamError as err:
        sys.exit(str(err))
    console_detail(args.verbose or str((config.get("system") or {}).get("console", "")).lower() == "verbose")

    if args.doctor:
        from core.diagnostics import doctor
        sys.exit(doctor(config))
    if args.stop:
        sys.exit(stop_listener(config))
    if args.mic_check:
        sys.exit(mic_check(config))
    if args.test_voice:
        test_voices(config)
    elif args.cli:
        target, prompt = args.cli
        assistants_cfg = config.get("assistants", {})
        dispatcher = AdapterDispatcher(assistants_cfg)
        names = {k: a.get("name", k.title()) for k, a in assistants_cfg.items()}
        group_key = f"group:{team.slug(target)}"
        if target.lower() == "team" or group_key in (config.get("groups") or {}):
            council_cfg = config.get("council", {}) or {}
            members = [m for m in (council_cfg.get("members") or list(assistants_cfg)) if m in dispatcher.adapters]
            if target.lower() != "team":
                members = [m for m in config["groups"][group_key]["members"] if m in dispatcher.adapters]
            members = [m for m in members if not dispatcher.adapters[m].unavailable()]
            for member, reply in run_council(prompt, members, names, dispatcher.dispatch, lambda m: "",
                                             parallel=bool(council_cfg.get("parallel", True)),
                                             reply_order=council_cfg.get("reply_order", "completion"),
                                             roles={m: dispatcher.adapters[m].role for m in members}):
                show_reply(names.get(member, member), reply)
        else:
            key = target.lower()
            if key not in dispatcher.adapters:
                found = team.find([{"name": a.get("name", k), "kind": k} for k, a in assistants_cfg.items()], target)
                key = found["kind"] if found else key
            adapter = dispatcher.adapters.get(key)
            if adapter is None:
                sys.exit(f"There is no team member called {target}. On the team: {', '.join(names.values())}.")
            resp, voice, speaker = dispatcher.dispatch(key, build_prompt(
                prompt, key, names.get(key, key), "", [],
                hint=not getattr(adapter, "spoken_system", False), role=adapter.role))
            show_reply(names.get(key, key), resp)
    else:
        # Plain `mcbeak` listens. Checked before any model loads or the
        # microphone opens; the lock is held until exit.
        try:
            status_port = port_from(args.status_port)
        except ValueError as err:
            sys.exit(str(err))
        lock = claim_listener(config, replace=args.replace)  # noqa: F841 (kept open)
        run_voice_daemon(config, use_tmux=args.tmux, config_path=args.config, status_port=status_port)


if __name__ == "__main__":
    main()
