"""
Turn the conversation log into a who-said-what script.

The listener writes one log on this machine (see log_path in mcbeak.py). This
module reads it, nothing else: no model, no microphone, no network. A session
runs from one "Starting <name>..." line to the next.

Lines it reads:
    Registered assistant adapter: Claude Code (claude)   member key -> name
    STT clip: ...                                        a new utterance begins
    Tone: happy 0.76, angry 0.18, ...                    tone of that utterance
    🎙️  Heard (0.30s): "..."                             what was said
    🎯 Matched Target: [CLAUDE] ...                      who it was said to
    claude listening (30s follow-up window, ...)         a name on its own
    Dispatching prompt to [Grok]: 'You are Grok, one voice in a spoken group...'
    Speaking (claude): '...'                             a reply, in full
    Speaking (claude) as the reply arrives               a reply spoken as written
    Spoke (claude) as the reply arrived: '...'           its text, once complete
"""
import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \[(\w+)\] (.*)$")
_START = re.compile(r"^Starting (.+)\.\.\.$")
_ADAPTER = re.compile(r"^Registered assistant adapter: (.+) \(([^()]+)\)$")
_HEARD = re.compile(r'Heard \([\d.]+s\): "(.*)"$')
_MATCHED = re.compile(r"Matched Target: \[(.+?)\]")
_LISTENING = re.compile(r"^(\S+) listening \(")
_TONE = re.compile(r"^Tone: (\w+) ([\d.]+)")
_GROUP_DISPATCH = re.compile(r"^Dispatching prompt to \[.+?\]: 'You are .+, one voice in a spoken group")
_SPEAKING = re.compile(r"^Speaking \((.+?)\): (.*)$")
_STREAM_START = re.compile(r"^Speaking \((.+?)\) as the reply arrives$")
_STREAM_DONE = re.compile(r"^Spoke \((.+?)\) as the reply arrived(, interrupted)?: (.*)$")
_FAILED = re.compile(r"^(.+?) (?:could not answer and will sit out|did not answer\.$)")

# Replies were once logged cut to this many characters
_OLD_CUT = 120
# A name said on its own keeps the microphone open for the question this long
_FOLLOW_UP_SECONDS = 45
_NOT_LOGGED = "reply spoken as it arrived; its text was not logged"
_QUIET_TONES = ("neutral", "unknown", "other")


@dataclass
class Entry:
    time: datetime
    kind: str                      # "you", "reply" or "note"
    speaker: str
    text: str
    to: str = ""                   # who "you" lines were said to; "" when to nobody
    tone: str = ""
    tone_score: float = 0.0
    round: Optional[int] = None    # team rounds: the question and its answers share a number
    interrupted: bool = False
    wake_only: bool = field(default=False, repr=False)
    member: str = field(default="", repr=False)


@dataclass
class Session:
    number: int
    name: str
    start: datetime
    end: datetime
    records: List[Tuple[datetime, str, str]]


# -- reading ---------------------------------------------------------------
def parse_line(line: str) -> Optional[Tuple[datetime, str, str]]:
    """(time, level, message) for a log line; None for a continuation line."""
    found = _LINE.match(line.rstrip("\n"))
    if not found:
        return None
    return datetime.strptime(found.group(1), "%Y-%m-%d %H:%M:%S"), found.group(2), found.group(3)


def log_files(path: Path) -> List[Path]:
    """The log and its rotated copies, oldest first."""
    older = sorted((p for p in path.parent.glob(path.name + ".*") if p.suffix[1:].isdigit()),
                   key=lambda p: -int(p.suffix[1:]))
    return [p for p in older + [path] if p.is_file()]


def read_records(path: Path) -> List[Tuple[datetime, str, str]]:
    records = []
    for file in log_files(path):
        with open(file, "r", encoding="utf-8", errors="replace") as handle:
            records.extend(r for r in map(parse_line, handle) if r)
    return records


def split_sessions(records: Iterable[Tuple[datetime, str, str]]) -> List[Session]:
    """One session per "Starting <name>..." line. Lines before the first one
    (a log that rotated mid-session) belong to no session and are dropped."""
    sessions: List[Session] = []
    for record in records:
        started = _START.match(record[2])
        if started:
            carried = []
            if sessions and len(sessions[-1].records) > 1 and sessions[-1].records[-1][0] == record[0] \
                    and sessions[-1].records[-1][2].startswith("Status endpoint on"):
                # Logged just before the start line; it belongs to the session that is starting
                carried = [sessions[-1].records.pop()]
                sessions[-1].end = sessions[-1].records[-1][0]
            sessions.append(Session(len(sessions) + 1, started.group(1), record[0], record[0], carried))
        if sessions:
            sessions[-1].records.append(record)
            sessions[-1].end = record[0]
    return sessions


def _unquote(text: str) -> str:
    """The string behind a logged repr; the text as written when it is not one."""
    try:
        value = ast.literal_eval(text)
        return value if isinstance(value, str) else text
    except (ValueError, SyntaxError):
        return text.strip("'\"")


# -- building the script ---------------------------------------------------
def build_entries(records: Iterable[Tuple[datetime, str, str]]) -> List[Entry]:
    names: Dict[str, str] = {}
    entries: List[Entry] = []
    said: Optional[Entry] = None         # the utterance being handled
    block_said: Optional[Entry] = None   # the utterance of the audio clip in hand
    block_tone: Optional[Tuple[str, float]] = None
    streaming: Dict[str, Entry] = {}
    round_no = 0

    def name_of(key: str) -> str:
        return names.get(key) or ("Team" if key == "team" else key if " " in key or key[:1].isupper() else key.title())

    def reply(when: datetime, key: str, text: str) -> Entry:
        speaker = name_of(key)
        if text.startswith(f"{speaker} says: "):
            text = text[len(speaker) + 7:]
        failed = _FAILED.match(text)
        entry = Entry(when, "note" if failed else "reply", speaker,
                      "could not answer" if failed else text, member=key,
                      round=said.round if said else None)
        entries.append(entry)
        return entry

    for when, _level, message in records:
        if message.startswith("STT clip:"):
            block_said, block_tone = None, None
            continue
        found = _ADAPTER.match(message)
        if found:
            names[found.group(2)] = found.group(1)
            continue
        found = _TONE.match(message)
        if found:
            tone = (found.group(1), float(found.group(2)))
            if block_said is not None and not block_said.tone:
                block_said.tone, block_said.tone_score = tone
            else:
                block_tone = tone
            continue
        found = _HEARD.search(message)
        if found:
            said = block_said = Entry(when, "you", "You", found.group(1))
            if block_tone:
                said.tone, said.tone_score = block_tone
            entries.append(said)
            continue
        if said is not None:
            found = _MATCHED.search(message)
            if found and not said.to:
                said.member = found.group(1).lower()
                said.to = name_of(said.member)
                continue
            found = _LISTENING.match(message)
            if found and found.group(1) == said.member:
                said.wake_only = True
                continue
            if _GROUP_DISPATCH.match(message) and said.round is None:
                round_no += 1
                said.round = round_no
                continue
        found = _STREAM_START.match(message)
        if found:
            streaming[found.group(1)] = reply(when, found.group(1), "")
            continue
        found = _STREAM_DONE.match(message)
        if found:
            entry = streaming.pop(found.group(1), None) or reply(when, found.group(1), "")
            entry.text, entry.interrupted = _unquote(found.group(3)), bool(found.group(2))
            continue
        found = _SPEAKING.match(message)
        if found:
            text = _unquote(found.group(2))
            reply(when, found.group(1), text + "…" if len(text) == _OLD_CUT else text)
    for entry in entries:
        if entry.kind == "reply" and not entry.text:
            entry.kind, entry.text = "note", _NOT_LOGGED
    return join_wake_words(entries)


def join_wake_words(entries: List[Entry]) -> List[Entry]:
    """"Hey team." followed by the question is one thing said, not two."""
    out: List[Entry] = []
    for entry in entries:
        last = out[-1] if out else None
        if (last is not None and last.kind == "you" and last.wake_only and entry.kind == "you"
                and entry.member == last.member
                and (entry.time - last.time).total_seconds() <= _FOLLOW_UP_SECONDS):
            last.text = f"{last.text} {entry.text}"
            last.wake_only, last.round = entry.wake_only, entry.round
            if entry.tone:
                last.tone, last.tone_score = entry.tone, entry.tone_score
            continue
        out.append(entry)
    return out


def parse_when(text: str, day: datetime) -> datetime:
    """"21:54" or "21:54:30" on the given day, or a full ISO date and time."""
    text = text.strip()
    for shape in ("%H:%M", "%H:%M:%S"):
        try:
            clock = datetime.strptime(text, shape)
            return day.replace(hour=clock.hour, minute=clock.minute, second=clock.second, microsecond=0)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError:
        raise ValueError(f'"{text}" is not a time. Use HH:MM or a date and time like 2026-01-31T21:54.')


def select(entries: List[Entry], since: Optional[datetime] = None, until: Optional[datetime] = None,
           unaddressed: bool = False) -> List[Entry]:
    """Entries inside the time range; speech said to nobody only when asked for."""
    return [e for e in entries
            if (since is None or e.time >= since) and (until is None or e.time <= until)
            and (unaddressed or e.kind != "you" or e.to)]


# -- writing ---------------------------------------------------------------
def _tone(entry: Entry, tone: bool) -> str:
    return entry.tone if tone and entry.tone and entry.tone not in _QUIET_TONES else ""


def _aside(entry: Entry, tone: bool) -> str:
    parts = [f"to {entry.to}" if entry.to else "to nobody by name", _tone(entry, tone) and f"tone: {_tone(entry, tone)}"]
    return "; ".join(p for p in parts if p)


def span(start: datetime, end: datetime) -> str:
    last = end.strftime("%H:%M:%S") if end.date() == start.date() else end.strftime("%Y-%m-%d %H:%M:%S")
    return f"{start.strftime('%Y-%m-%d %H:%M:%S')} to {last}"


def render_markdown(entries: List[Entry], title: str, label: str, tone: bool = True) -> str:
    lines = [f"# {title} transcript", "", label, ""]
    for i, entry in enumerate(entries):
        clock = entry.time.strftime("%H:%M:%S")
        cut = " *(interrupted)*" if entry.interrupted else ""
        in_round = entry.round is not None and entry.kind != "you"
        if entry.kind == "you":
            body = f"**{clock} You** ({_aside(entry, tone)}): {entry.text}"
        elif entry.kind == "note":
            body = f"**{clock} {entry.speaker}:** *{entry.text}*"
        else:
            body = f"**{clock} {entry.speaker}:** {entry.text}{cut}"
        lines.append(f"- {body}" if in_round else body)
        following = entries[i + 1] if i + 1 < len(entries) else None
        if not (in_round and following is not None and following.round == entry.round and following.kind != "you"):
            lines.append("")
    if not entries:
        lines += ["*Nothing was said.*", ""]
    return "\n".join(lines)


def render_text(entries: List[Entry], title: str, label: str, tone: bool = True) -> str:
    lines = [f"{title} transcript", label, ""]
    for entry in entries:
        clock = entry.time.strftime("%H:%M:%S")
        indent = "    " if entry.round is not None and entry.kind != "you" else ""
        if entry.kind == "you":
            lines.append(f"[{clock}] You ({_aside(entry, tone)}): {entry.text}")
        elif entry.kind == "note":
            lines.append(f"{indent}[{clock}] {entry.speaker}: ({entry.text})")
        else:
            lines.append(f"{indent}[{clock}] {entry.speaker}: {entry.text}{' (interrupted)' if entry.interrupted else ''}")
    return "\n".join(lines) + "\n"


def _srt_clock(seconds: float) -> str:
    whole, millis = divmod(int(round(max(seconds, 0.0) * 1000)), 1000)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d},{millis:03d}"


def render_srt(entries: List[Entry], origin: datetime) -> str:
    """Captions timed from origin. The log keeps whole seconds and no end times,
    so a caption lasts as long as its words take to read (1.5 to 8 s) or until
    the next one starts. Notes are left out."""
    spoken = [e for e in entries if e.kind != "note"]
    blocks = []
    for i, entry in enumerate(spoken):
        start = (entry.time - origin).total_seconds()
        end = start + min(8.0, max(1.5, 0.4 * len(entry.text.split())))
        if i + 1 < len(spoken):
            following = (spoken[i + 1].time - origin).total_seconds()
            if following > start:
                end = min(end, following)
        blocks.append(f"{i + 1}\n{_srt_clock(start)} --> {_srt_clock(end)}\n{entry.speaker}: {entry.text}\n")
    return "\n".join(blocks)


def render_json(entries: List[Entry], title: str, label: str, tone: bool = True) -> str:
    rows = []
    for entry in entries:
        row = asdict(entry)
        row.pop("wake_only")
        row.pop("member")
        row["time"] = entry.time.isoformat(sep=" ")
        if not tone:
            row.pop("tone")
            row.pop("tone_score")
        rows.append(row)
    return json.dumps({"name": title, "session": label, "entries": rows}, indent=2, ensure_ascii=False) + "\n"


def list_sessions(sessions: List[Session]) -> str:
    lines = []
    for session in sessions:
        entries = select(build_entries(session.records))
        said = next((e.text for e in entries if e.kind == "you"), "")
        first = f'  "{said[:60]}{"…" if len(said) > 60 else ""}"' if said else ""
        lines.append(f"{session.number:>4}  {session.start.strftime('%Y-%m-%d %H:%M:%S')} to "
                     f"{session.end.strftime('%H:%M:%S')}  {len(entries):>3} lines{first}")
    return "\n".join(lines) + "\n"


# -- command ---------------------------------------------------------------
def main(argv: List[str], log: Path) -> int:
    parser = argparse.ArgumentParser(
        prog="mcbeak transcript",
        description="Write out a session as a who-said-what script, from the log on this machine.")
    parser.add_argument("--session", default="last", metavar="last|N|all|list",
                        help="Which session: the last one (default), number N from `--session list`, or all of the log")
    parser.add_argument("--since", metavar="TIME", help='Start at this time: "21:54" (on the session\'s day) or 2026-01-31T21:54')
    parser.add_argument("--until", metavar="TIME", help="Stop at this time (same forms)")
    parser.add_argument("--format", choices=["md", "txt", "srt", "json"], default="md",
                        help="md (default), txt, srt (captions timed from the session start, or from --since) or json")
    parser.add_argument("--out", metavar="PATH", help="Write to this file instead of the terminal")
    parser.add_argument("--no-tone", action="store_true", help="Leave out the tone of voice")
    parser.add_argument("--all", action="store_true", dest="unaddressed",
                        help="Include speech that was heard but said to nobody by name")
    parser.add_argument("--log", type=Path, default=log, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    sessions = split_sessions(read_records(args.log))
    if not sessions:
        print(f"No sessions in {args.log} yet.", file=sys.stderr)
        return 1
    if args.session == "list":
        sys.stdout.write(list_sessions(sessions))
        return 0
    if args.session == "all":
        chosen = sessions
    elif args.session == "last":
        chosen = sessions[-1:]
    elif args.session.isdigit() and 1 <= int(args.session) <= len(sessions):
        chosen = [sessions[int(args.session) - 1]]
    else:
        print(f'There is no session "{args.session}". `mcbeak transcript --session list` shows 1 to {len(sessions)}.',
              file=sys.stderr)
        return 1

    start, end, title = chosen[0].start, chosen[-1].end, chosen[-1].name
    day = start if len(chosen) == 1 else datetime.now()
    try:
        since = parse_when(args.since, day) if args.since else None
        until = parse_when(args.until, day) if args.until else None
    except ValueError as err:
        print(err, file=sys.stderr)
        return 1
    if until is not None and args.until and re.fullmatch(r"\d\d?:\d\d", args.until.strip()):
        until += timedelta(seconds=59)   # "21:56" takes in the whole minute
    entries = select([e for s in chosen for e in build_entries(s.records)], since, until, args.unaddressed)
    tone = not args.no_tone
    if args.format == "srt":
        text = render_srt(entries, since or start)
    else:
        render = {"md": render_markdown, "txt": render_text, "json": render_json}[args.format]
        which = f"Session {chosen[0].number}" if len(chosen) == 1 else f"Sessions 1 to {len(chosen)}"
        label = f"{which}, {span(max(start, since) if since else start, min(end, until) if until else end)}"
        text = render(entries, title, label, tone)
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0
