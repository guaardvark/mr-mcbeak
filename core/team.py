"""
The team: who is in the room.

A member is a name plus a kind (claude, codex, grok, antigravity, cursor,
guaardvark, ollama). The team is kept in one small file the user owns,

    ~/.config/mcbeak/team.yaml        (or $MCBEAK_TEAM)

    team:
      - name: Claude
        kind: claude
      - name: Ada
        kind: claude
        folder: ~/projects/site
        model: claude-haiku-5-5
        role: You review changes before they land.
      - name: Gemma
        kind: ollama
        model: gemma4:12b

and everything else is filled in: wake words from the name, a voice nobody
else on the team has, and the kind's settings from config.yaml. Without the
file, the team is every assistant in config.yaml whose program is installed.

The file is the whole interface: `mcbeak team add/remove` edit it, a running
copy reloads it when it changes, and any other program (a Guaardvark page,
another machine syncing the file) can manage the team the same way.
"""
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from core.wake_word_matcher import levenshtein_distance

KINDS = ("claude", "codex", "grok", "antigravity", "cursor", "guaardvark", "ollama")
KIND_ALIASES = {"agy": "antigravity", "gemini": "antigravity", "local": "ollama"}
# What a kind needs installed, said when it is missing
KIND_PROGRAM = {"claude": "claude", "codex": "codex", "grok": "grok", "antigravity": "agy", "cursor": "agent"}
# Kinds that send what you say to them to a company's servers
CLOUD_KINDS = ("claude", "codex", "grok", "antigravity", "cursor")
# Settings a member takes from its kind's entry in config.yaml
KIND_KEYS = ("cli_cmd", "mode", "mcp", "api_url", "use_rag", "effort", "confirm_first", "timeout_seconds")
# Settings that belong to the member the config.yaml entry is named after
IDENTITY_KEYS = ("wake_words", "phonetic_variants", "voice", "voice_speaker", "description")
# What a member in the team file may set
MEMBER_KEYS = ("name", "kind", "folder", "workspace", "model", "permissions", "role", "voice", "voice_speaker",
               "wake_words", "phonetic_variants", "host", "mcp", "effort", "timeout_seconds", "think",
               "api_url", "use_rag", "history_messages", "confirm_first", "cli_cmd")

# Voices handed out to members that do not name one. The first ten were heard
# clearly by Whisper when measured (64-65 of 67 words); the rest are unmeasured.
KOKORO_VOICES = ("am_michael", "bm_george", "am_adam", "bm_lewis", "am_echo", "bm_daniel", "am_eric",
                 "bm_fable", "am_liam", "am_onyx", "af_heart", "bf_emma", "af_bella", "bf_alice",
                 "af_nicole", "af_sarah", "bf_lily", "af_nova")
PIPER_FALLBACK = ("en_US-libritts-high.onnx", (430, 3, 24, 42, 12, 60, 101, 150, 200, 250))

# Words a member cannot be called: they already mean something when spoken
RESERVED = {"team", "everyone", "everybody", "stop", "sleep", "repeat", "yes", "no", "yeah", "nope", "okay",
            "ok", "hey", "hi", "hello", "yo", "cancel", "wait", "sure", "fine", "new", "add", "remove"}


class TeamError(ValueError):
    """Something about the team the user has to fix; the message says what."""


def team_path() -> Path:
    if os.environ.get("MCBEAK_TEAM"):
        return Path(os.path.expanduser(os.environ["MCBEAK_TEAM"]))
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "mcbeak" / "team.yaml"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def spoken(name: str) -> str:
    """A name as the wake-word matcher sees it: lower case, letters and digits."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(name).lower()).split())


def kind_of(key: str, entry: dict) -> str:
    kind = str(entry.get("kind") or entry.get("target_id") or key).lower()
    return KIND_ALIASES.get(kind, kind)


def templates(config: dict) -> Dict[str, Tuple[str, dict]]:
    """kind -> (key, entry) of the config.yaml assistant that describes it."""
    found: Dict[str, Tuple[str, dict]] = {}
    for key, entry in (config.get("assistants") or {}).items():
        found.setdefault(kind_of(key, entry), (key, entry))
    return found


def read_team(path: Optional[Path] = None) -> Optional[List[dict]]:
    """The members in the team file, or None when there is no file."""
    path = path or team_path()
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as err:
        raise TeamError(f"{path} is not valid YAML: {err}")
    members = data.get("team") if isinstance(data, dict) else data
    if not isinstance(members, list) or not all(isinstance(m, dict) for m in members):
        raise TeamError(f"{path} must hold a list of members under 'team:'")
    return members


def write_team(members: List[dict], path: Optional[Path] = None):
    path = path or team_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ("# Who is on the team. Edit here or with `mcbeak team add/remove`;\n"
            "# a running copy picks the change up by itself.\n"
            f"# Kinds: {', '.join(KINDS)}\n")
    body = yaml.safe_dump({"team": members}, sort_keys=False, allow_unicode=True, default_flow_style=False)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(head + body, encoding="utf-8")
    os.replace(tmp, path)


def _own_name(member_name: str, key: str, entry: dict) -> bool:
    """Whether a member is the one a config.yaml entry is named after
    ("Claude", "claude" and "Claude Code" are all that entry's own member)."""
    names = {slug(key), slug(entry.get("name", "")), slug(kind_of(key, entry))}
    names |= {slug(w) for w in (entry.get("wake_words") or [])[:1]}
    return slug(member_name) in names - {""}


def default_wake_words(name: str) -> List[str]:
    said = spoken(name)
    return [said, f"hey {said}", f"ok {said}", f"okay {said}", f"hello {said}"]


def _sounds_like(a: str, b: str) -> bool:
    a, b = a.replace(" ", ""), b.replace(" ", "")
    if not a or not b:
        return False
    distance = levenshtein_distance(a, b)
    if max(len(a), len(b)) <= 4:
        return distance <= 1
    return 1.0 - distance / max(len(a), len(b)) >= 0.7


def check_name(name: str, taken: Dict[str, List[str]], config: dict):
    """Refuse a name that could not be told apart by ear. `taken` maps each
    current member's name to the phrases it answers to."""
    said = spoken(name)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 .'-]{1,23}", name.strip()) or not said:
        raise TeamError(f'"{name}" cannot be a name: use 2 to 24 letters, digits or spaces, starting with a letter.')
    if set(said.split()) & RESERVED:
        raise TeamError(f'"{name}" cannot be a name: it contains a word that is already a spoken command.')
    system_words = {spoken(bare) for bare in [(config.get("system") or {}).get("name", "")]
                    + list(config.get("coordinator_wake_words") or [])
                    + list((config.get("council") or {}).get("wake_words") or [])}
    for word in system_words - {""}:
        if _sounds_like(said, re.sub(r"^(hey|ok|okay|hi|hello|yo) ", "", word)):
            raise TeamError(f'"{name}" sounds too much like "{word}", which the app itself answers to.')
    for other, phrases in taken.items():
        for phrase in phrases:
            bare = re.sub(r"^(hey|ok|okay|hi|hello|yo) ", "", spoken(phrase))
            if _sounds_like(said, bare):
                raise TeamError(f'"{name}" sounds too much like "{bare}", which {other} answers to. '
                                "Pick a name that is easy to tell apart by ear.")


def _voice_pool(root: Path) -> List[Tuple[str, int]]:
    kokoro = root / "models" / "kokoro"
    if (kokoro / "kokoro-v1.0.onnx").exists() and (kokoro / "voices-v1.0.bin").exists():
        return [(f"kokoro:{v}", 0) for v in KOKORO_VOICES]
    return [(PIPER_FALLBACK[0], n) for n in PIPER_FALLBACK[1]]


def resolve(members: List[dict], config: dict, root: Optional[Path] = None) -> Dict[str, dict]:
    """Team-file members as full assistant entries, keyed by member."""
    root = root or Path(__file__).resolve().parent.parent
    kinds = templates(config)
    out: Dict[str, dict] = {}
    taken: Dict[str, List[str]] = {}
    for raw in members:
        name = str(raw.get("name") or "").strip()
        if not name:
            raise TeamError("A team member has no name.")
        kind = KIND_ALIASES.get(str(raw.get("kind") or slug(name)).lower(), str(raw.get("kind") or slug(name)).lower())
        if kind not in KINDS:
            raise TeamError(f'{name}: unknown kind "{raw.get("kind")}". Kinds: {", ".join(KINDS)}.')
        unknown = [k for k in raw if k not in MEMBER_KEYS]
        if unknown:
            raise TeamError(f"{name}: unknown setting {', '.join(unknown)}.")
        tkey, tentry = kinds.get(kind, (kind, {}))
        own = bool(tentry) and _own_name(name, tkey, tentry) and tkey not in out
        key = tkey if own else slug(name)
        if key in out or key in ("team", "coordinator"):
            raise TeamError(f'Two team members are called "{name}".')
        entry = {k: tentry[k] for k in KIND_KEYS if k in tentry}
        if own:
            entry.update({k: tentry[k] for k in IDENTITY_KEYS if k in tentry})
            entry["name"] = tentry.get("name", name)
        else:
            entry["name"] = name
            check_name(name, taken, config)
        # Guaardvark has no read-only mode to enforce, so read-only would mean
        # it refuses every request; it runs on the user's own machine
        entry["permissions"] = "full" if kind == "guaardvark" else "readonly"
        entry.update({k: v for k, v in raw.items() if k not in ("name", "kind") and v is not None})
        entry["kind"] = kind
        entry.setdefault("wake_words", default_wake_words(name))
        taken[entry["name"]] = list(entry["wake_words"]) + list(entry.get("phonetic_variants") or [])
        out[key] = entry
    # Voices last, so a member that names one keeps it and the rest avoid it
    used = {(e["voice"], int(e.get("voice_speaker", 0))) for e in out.values() if e.get("voice")}
    pool = [v for v in _voice_pool(root) if v not in used]
    for entry in out.values():
        if not entry.get("voice"):
            voice, speaker = pool.pop(0) if pool else _voice_pool(root)[0]
            entry["voice"], entry["voice_speaker"] = voice, speaker
        entry.setdefault("voice_speaker", 0)
    return out


def missing_program(entry: dict) -> Optional[str]:
    """The program a member needs that is not installed here, or None."""
    program = KIND_PROGRAM.get(entry.get("kind") or "")
    if not program:
        return None
    names = [entry.get("cli_cmd") or program] + (["cursor-agent"] if program == "agent" else [])
    return None if any(shutil.which(n) for n in names) else names[0]


def apply(config: dict, path: Optional[Path] = None, root: Optional[Path] = None) -> dict:
    """Put the team into `config`: `assistants` becomes the members that can
    run here, the council and the default assistant follow, and
    config["team_notes"] lists who was left out and why."""
    members = read_team(path)
    notes = []
    if members is None:
        team = {}
        for key, entry in (config.get("assistants") or {}).items():
            team[key] = dict(entry, kind=kind_of(key, entry))
    else:
        team = resolve(members, config, root)
    present = {}
    for key, entry in team.items():
        program = missing_program(entry)
        if program:
            notes.append(f"{entry.get('name', key)} is not on the team here: the {program} command is not installed.")
        else:
            present[key] = entry
    if not present:
        raise TeamError("Nobody on the team can run on this machine. " + " ".join(notes)
                        + " Add someone with: mcbeak team add NAME KIND")
    config["assistants"] = present
    config["team_notes"] = notes
    # Names of the assistants config.yaml describes, given to speech recognition
    # so one that is not on the team yet can still be asked to join by voice
    config["known_names"] = [" ".join(w.capitalize() for w in spoken((e.get("wake_words") or [k])[0]).split())
                             for k, e in (config.get("assistants") or {}).items()]
    config["team_file"] = str(path or team_path()) if members is not None else ""
    council = dict(config.get("council") or {})
    listed = council.get("members") if members is None else None
    council["members"] = [m for m in (listed or list(present)) if m in present] or list(present)
    config["council"] = council
    system = dict(config.get("system") or {})
    aliases = set(present) | {e.get("target_id") for e in present.values()}
    if system.get("default_assistant", "claude") not in aliases:
        system["default_assistant"] = next(iter(present))
    config["system"] = system
    return config


def current_members(config: dict, path: Optional[Path] = None) -> List[dict]:
    """The team as a list for the team file: the file's own members, or, before
    the file exists, the installed assistants from config.yaml."""
    members = read_team(path)
    if members is not None:
        return members
    out = []
    for key, entry in (config.get("assistants") or {}).items():
        kind = kind_of(key, entry)
        if missing_program(dict(entry, kind=kind)):
            continue
        member = {"name": entry.get("name") or key.title(), "kind": kind}
        for k in ("workspace", "model", "permissions", "role"):
            if entry.get(k):
                member[k] = entry[k]
        out.append(member)
    return out


def find(members: List[dict], name: str) -> Optional[dict]:
    wanted = slug(name)
    for member in members:
        if slug(member.get("name", "")) == wanted:
            return member
    # "Claude" finds the member called "Claude Code"; failing that, a kind's only member
    for close in ([m for m in members if slug(m.get("name", "")).startswith(wanted + "_")],
                  [m for m in members if slug(m.get("kind", "")) == wanted]):
        if len(close) == 1:
            return close[0]
    return None


def add_member(config: dict, name: str, kind: Optional[str] = None, path: Optional[Path] = None,
               root: Optional[Path] = None, **settings) -> dict:
    """Add a member to the team file (creating it from the current team first).
    Returns the member's resolved entry. Raises TeamError with the reason when
    the name, kind or settings will not work."""
    members = [dict(m) for m in current_members(config, path)]
    kind = KIND_ALIASES.get(str(kind or slug(name)).lower(), str(kind or slug(name)).lower())
    if kind not in KINDS:
        raise TeamError(f'Say what kind of assistant "{name}" is: {", ".join(KINDS)}.')
    if any(slug(m.get("name", "")) == slug(name) for m in members):
        raise TeamError(f"{name} is already on the team.")
    member = {"name": name.strip(), "kind": kind}
    member.update({k: v for k, v in settings.items() if v not in (None, "")})
    if member.get("permissions") not in (None, "readonly", "full"):
        raise TeamError("permissions must be readonly or full.")
    folder = member.get("folder")
    if folder and not Path(os.path.expanduser(folder)).is_dir():
        raise TeamError(f"The folder {folder} does not exist.")
    program = missing_program({"kind": kind, "cli_cmd": (templates(config).get(kind) or ("", {}))[1].get("cli_cmd")})
    if program:
        raise TeamError(f"{name} needs the {program} command, which is not installed on this machine. "
                        "Install it, sign in, then add the member again.")
    members.append(member)
    resolved = resolve(members, config, root)   # validates the whole team
    write_team(members, path)
    return list(resolved.values())[-1]


def remove_member(config: dict, name: str, path: Optional[Path] = None) -> dict:
    members = [dict(m) for m in current_members(config, path)]
    member = find(members, name)
    if member is None:
        names = ", ".join(m.get("name", "?") for m in members) or "nobody"
        raise TeamError(f"There is no team member called {name}. On the team: {names}.")
    if len(members) == 1:
        raise TeamError(f"{member['name']} is the only one on the team; add someone else first.")
    members.remove(member)
    write_team(members, path)
    return member
