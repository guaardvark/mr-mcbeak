# Mr. McBeak

Hands-free voice front end for several AI coding assistants on one machine:
Claude Code, Codex, Grok, Antigravity, Cursor Agent, a local Guaardvark and any
model served by Ollama. Speech is recognised locally (faster-whisper) and replies
are spoken locally (Kokoro or Piper), each assistant in its own voice. You choose
who is on the team, from one member to as many as you can tell apart by ear.

## Run

```
./setup.sh            # venv, Kokoro and Piper voices, models, ~/.local/bin/mcbeak symlink
mcbeak                # always-on listener (same as mcbeak --listen)
mcbeak --verbose      # same, with the technical log lines in the terminal too
mcbeak --replace      # stop the copy already listening and start this one
mcbeak --stop         # stop the copy that is listening
mcbeak --test-voice   # hear every assistant's voice
mcbeak --mic-check    # measure room noise, the reply's echo and your voice
mcbeak --cli grok "what is a decorator"        # one text turn, no microphone
mcbeak --cli team "which database should we use"   # ask everyone
mcbeak team           # who is on the team
mcbeak team add Grok  # add a member; see "The team"
mcbeak transcript     # the last session as a who-said-what script; see "Transcripts"
```

The terminal shows the conversation: what you said and to whom, each reply,
and a dim line for speech it heard without a name. Only one copy listens at a
time: a second one names the copy already running (PID, start time, terminal)
and exits before it opens the microphone. Ctrl+C or Ctrl+Z quits (Ctrl+Z would
otherwise leave it paused, still holding the microphone). Each session is logged to `~/.local/state/mcbeak/mcbeak.log`, which
includes what was heard and stays on this machine.

## The team

A member is a name plus a kind: `claude`, `codex`, `grok`, `antigravity`,
`cursor`, `guaardvark` or `ollama`. Until you change it, the team is every
assistant in `config.yaml` whose program is installed on the machine; one that
is missing is left out and named when the app starts.

```
mcbeak team                                    # name, kind, model, folder, voice, permissions
mcbeak team add Grok                           # the name is the kind
mcbeak team add Ada claude --folder ~/site --model claude-haiku-5-5 \
    --role "You review changes before they land."
mcbeak team add Gemma ollama --model gemma4:12b   # a model on your own machine
mcbeak team remove Ada
```

The first `add` or `remove` writes the team to `~/.config/mcbeak/team.yaml`
(`$MCBEAK_TEAM` overrides the path), and from then on that file is the team.
It holds only what you chose; the rest is filled in:

- wake words from the name (`Ada`, `Hey Ada`, `OK Ada`), and the name is given
  to Whisper so it is heard;
- a voice nobody else on the team has. The first ten handed out are the Kokoro
  voices Whisper understood best when measured; a machine without Kokoro gets
  Piper LibriTTS speakers;
- `permissions: readonly` unless you pass `--permissions full`.

A name is refused when it could be mistaken for another member, for the app's
own name, or for a spoken command: `Grock` next to Grok, `Aida` next to Ada,
`Stop`. Two members of one kind each keep their own conversation, so three
Claudes in three folders are three separate sessions.

A copy that is listening notices the file change within half a second: the new
member answers to its name without a restart and says so in its own voice.
Editing the file by hand, or having another program write it, works the same
way.

By voice, to the app's own name or to any member:

- `Mr. McBeak, who is on the team?`
- `Mr. McBeak, ask Grok to join.` It asks first ("Add Grok to the team? What
  you say to it goes to its maker's servers. Say yes or no.") and adds the
  member read-only on a spoken yes. This works for the kinds `config.yaml`
  describes; an Ollama model, a second member of a kind, a folder or full
  permissions go through `mcbeak team add`.
- `Mr. McBeak, ask Grok to leave.`

An `ollama` member talks to Ollama at `http://127.0.0.1:11434` (`--host` for
an Ollama on another of your machines). It has no tools and no file access,
its reply is spoken as it is written, and nothing said to it leaves your
machines. Its conversation is kept in memory and ends when the app stops.

With more than a few members, `Hey team` asks more assistants than you want
to wait for. A group is part of the team under its own name:

```
mcbeak team group reviewers Ada Grok     # "Hey reviewers, ..." asks Ada and Grok
mcbeak team group reviewers              # no members: the group is deleted
```

Groups are kept in the same file under `groups:`. A follow-up without a name
stays with the group, the same naming rules apply as for members, and a
member who leaves the team leaves its groups.

A member that cannot answer (program missing, not signed in, model not
pulled, no reply in time) says so once and then sits out `Hey team` questions.
Speaking to it by name tries it again, and one good answer puts it back.

## Talking to it

- `Hey Claude ...`, `Hey Codex ...`, `Hey Grok ...`, `Hey Antigravity ...`,
  `Hey Cursor ...`, `Hey Guaardvark ...` address one assistant. The wake name must come within the
  first three words; a name mentioned later in a sentence does not re-route.
- `Mr. McBeak ...` goes to the last assistant you spoke to, or the default.
  The names in `config.yaml` are given to Whisper (large-v3-turbo) as a lead-in
  sentence, so rare names like these are heard reliably: 39 of 40 spoken test
  commands reached the right assistant, and 1 of 200 unaddressed LibriSpeech
  clips woke one.
- `Hey Team ...` asks every member of the council. They answer one after another
  in their own voices. Parallel replies share prior context; sequential replies
  also see earlier answers from the current round.
- For 30 seconds after a reply you can keep talking without a wake word.
- What counts as speech in the live microphone is decided by Silero VAD
  (`audio.vad: silero`, the model faster-whisper ships, about 0.04 ms per
  32 ms of audio on one CPU thread) instead of a loudness level. LibriSpeech
  clips played into the recorder at a level of 0.006, below the old fixed
  level of 0.015, were captured 40 of 40 times against 0 of 40; synthetic door
  thumps started no recording against 3 in 20 seconds. `audio.vad: energy`
  brings the loudness level back.
- When you stop talking, a small local model (Smart Turn v3.2, CPU, about 20 ms)
  judges from how your last words sounded whether you finished. A finished
  question goes out half a second after your last word; when you sound
  mid-thought it waits up to three seconds (`audio.turn_*`, `max_silence_ms`).
  On pipecat's real-speech English test clips it caught 96% of finished turns
  and called 4.6% of unfinished ones finished.
- Anything you add while an assistant is still working on your question joins
  that question, so it answers both together instead of losing the first. The
  same goes for a pause mid-sentence: carry on within a few seconds
  (`audio.continuation_seconds`), before the reply has been spoken in full, and
  the reply stops and the whole sentence is asked again as one question.
- Claude's replies start playing as soon as the first sentence is written
  (`system.speak_early`); with Opus that was about 0.8 s sooner on a
  three-sentence answer. One-sentence answers play when they are complete.
- A Claude member's `claude` process stays up between questions
  (`keep_alive`), so a turn does not pay for starting it: 0.62 s against
  1.14 s on short questions with Haiku. Talking over a reply ends that process
  and the next question resumes the same session in a new one. A command you
  approved by voice runs in its own process, with only that command allowed.
  Each kept process holds about 220 MB.
- Tone of voice (`emotion.enabled`): a local model (emotion2vec+ large, about
  70 ms on the CPU, alongside transcription) listens to how you said it. A clear
  reading goes to the assistant with your words as a hint, such as "sounds
  annoyed or angry", and shows on screen next to what you said; neutral or
  unclear speech adds nothing. On the CREMA-D test set (actors, 1,117 clips) at
  the default threshold, 82% of emotional clips got the right hint and 7% of
  neutral ones got a hint. The hint leaves the machine with the words, which is
  why it is a switch, off by default; turning it on downloads about 650 MB.
- While an assistant works, a soft two-note pulse plays every few seconds
  (`system.thinking_cue`), so a long wait does not sound like a dead microphone.
- Each assistant's CLI session is resumed on every turn, so it remembers the
  conversation. The session ids are saved in
  `~/.local/state/mcbeak/sessions.json` (ids only, never the words), so a
  member carries on after a restart when it was used in the last 72 hours from
  the same folder (`system.remember_conversations`, `remember_hours`). `new
  conversation` forgets it.

## Transcripts

`mcbeak transcript` turns the log into a script of who said what and when. It reads
`~/.local/state/mcbeak/mcbeak.log` and its rotated copies and nothing else: the listener
does not have to be running, and nothing is sent anywhere.

```
mcbeak transcript                                   # the last session, Markdown, to the terminal
mcbeak transcript --session list                    # every session in the log, numbered
mcbeak transcript --session 12 --out notes.md       # session 12 to a file
mcbeak transcript --format srt --since 21:54 --out captions.srt
mcbeak transcript --session all --since 2026-01-31T09:00 --format json
```

A session runs from one start of the listener to the next. `--format` is `md`, `txt`,
`srt` or `json`. `--since` and `--until` take `HH:MM` (on the session's day) or a full
date and time. `--no-tone` drops the tone-of-voice note, and `--all` adds speech that
was heard but said to nobody by name.

```
**21:55:42 You** (to Team; tone: happy): Hey team. Everybody ready?

- **21:55:47 Cursor Agent:** *could not answer*
- **21:55:52 Guaardvark:** I'm ready! Let's get to work whenever you're set.
```

A name said on its own ("Hey team.") is joined to the question that follows it, and a
team round is listed under its question in the order the answers were spoken. SRT
captions count from the session start, or from `--since` when it is given, so a clip
that starts mid-session lines up. The log keeps whole seconds and no end times, so
each caption lasts as long as its words take to read, or until the next one starts.

Replies spoken while they were still being written are logged in full once they
finish. Sessions from before that change show such a reply as *reply spoken as it
arrived; its text was not logged*, and replies longer than 120 characters from those
sessions end in an ellipsis where the old log cut them.

## Safety

Supported adapters run read-only by default (Claude plan mode, Antigravity plan mode,
Cursor ask mode, Codex's read-only sandbox, Grok with write, edit and shell tools
denied). Set
`permissions: full` on an assistant in your settings file (see Config) to let
voice commands edit files or run commands. Remember the microphone hears everyone in the room.

With `permissions: full`, Claude Code runs in its auto mode: ordinary work goes
ahead, and its own classifier blocks destructive actions. Commands on the
assistant's `confirm_first` list (by default `git push`, `git reset --hard`,
`git clean`, `git checkout --`, `git restore`, `git branch -D`, `git stash drop`,
`git rm`, `rm` and `sudo`) are held as well. When something is held, Mr. McBeak
asks out loud, for example "Claude Code wants to run rm notes.txt. Should it go
ahead?". A spoken yes ("yes", "go ahead", "do it") re-runs the same session with
exactly that command allowed; "no" drops it; anything else is taken as a new
request.

## Telemetry

Speech recognition (faster-whisper) and the voices (Kokoro, Piper) run on this machine.
What leaves it is what you say to the assistant you name, sent by that
assistant's own CLI. Everything else is switched off in `core/privacy.py`, which
Mr. McBeak applies to itself before any library loads and to every CLI it starts:
ONNX Runtime usage events, Hugging Face pings (the Whisper model is read from
the local cache; the network is used once, to download it on first run), and
the CLIs' own usage metrics, crash reports, Grok's analytics and session-trace
upload, Codex's analytics, feedback and OpenTelemetry export (passed as `-c`
overrides on every call; traced: with them, a Codex turn contacts only
chatgpt.com), and Claude Code's update checks. Antigravity's usage telemetry is an
account setting, and Cursor's is governed by its Privacy Mode; neither has a
switch Mr. McBeak can set.

## Config

Your own choices go in `~/.config/mcbeak/settings.yaml` (`$MCBEAK_SETTINGS`
overrides the path), which is laid over `config.yaml` key by key, so updating
the checkout never undoes them:

```
emotion:
  enabled: true
assistants:
  claude:
    permissions: "full"
```

`config.yaml` holds the wake words, voices and the per-assistant keys
`permissions`, `model`, `workspace` and `timeout_seconds`. Claude also takes
`effort` (low to max) and `mcp`: `none` by default for voice turns, `all`, or a
list of your MCP server names. Loading MCP servers added about 1.5 s and $0.05 to
every short turn when measured, so add only the ones you want by voice. The `council` section
lists who answers to "Hey Team". STT model and device are under `stt`; on a
machine with a free GPU, `small.en` with `device: cuda` and `compute_type:
float16` is the first thing to try.

A voice is either a Piper model file (`en_US-libritts-high.onnx` with a
`voice_speaker` number) or a Kokoro voice (`kokoro:am_michael`). Kokoro sounds
more natural and Whisper understood it slightly better (65 of 67 words against
63 for Piper's LibriTTS #430), but its first sentence takes about 0.35 s to
synthesize on the CPU against 0.17 s. Clear male Kokoro voices: am_michael,
am_adam, am_echo, am_eric, am_liam, am_onyx, and British bm_george, bm_lewis,
bm_daniel, bm_fable. `mcbeak --test-voice` plays each assistant's voice.

## Models

All run on this machine; `./setup.sh` fetches them once, at pinned versions.

| Use | Model | Source | License |
|---|---|---|---|
| Speech recognition | Whisper large-v3-turbo, small.en (CTranslate2) | OpenAI; mobiuslabsgmbh and Systran conversions | MIT |
| End of turn | Smart Turn v3.2 | pipecat-ai/smart-turn-v3 | BSD-2-Clause |
| Tone of voice (off by default) | emotion2vec+ large (Ma et al., 2024), ONNX export | emotion2vec/emotion2vec_plus_large; Voldorn/emotion2vec-plus-large-onnx | FunASR Model Open Source License 1.1 |
| Voices | Kokoro v1.0 | hexgrad/Kokoro-82M via kokoro-onnx | Apache-2.0 |
| Voices | Piper LibriTTS | rhasspy/piper-voices | CC BY 4.0 |

## Tests

```
.venv/bin/python -m unittest discover tests
```

## Reliability and controls

The follow-up window starts after the reply finishes, including interrupted
replies. During an active conversation, say `stop`, `go to sleep`,
`new conversation`, or `repeat that`; "stop please", "be quiet", "start over"
and "say that again" are understood too. While passive, include a wake word.
New conversation resets the active assistant (all members for Team) and clears
shared history. Repeat replays the last spoken reply without another model call.

Interruptions cancel the current turn and suppress remaining council replies.
CLI process groups are terminated. A Guaardvark request already in progress is
abandoned at once, so the next question is not kept waiting, and its late
response is discarded; an Ollama reply stops generating. Backend
turns are serialized while transcription remains available for local controls.
Talking over a reply has to beat the reply's own echo. The app knows what it
is playing, learns from the first half second of playback how much of it
comes back into the microphone, and asks a voice to be `audio.echo_margin`
(2) times louder than the echo expected at that moment. In simulation
(replies echoed 120 ms late with a 150 ms room tail, 20 runs per level) no
reply interrupted itself with the echo between 0.006 and 0.04, where the old
fixed level interrupted every reply from 0.012 up, and a voice at 0.06 was
heard over the echo in 18 to 20 of 20 runs. This is a level comparison, not
echo cancellation: a quiet voice under loud speakers is still missed, and
headphones remove the problem. `mcbeak --mic-check` measures your room: the
noise floor, the echo of a spoken reply, your voice, and whether your voice
clears the bar.

Council requests run in parallel by default, using a snapshot of prior context.
`council.reply_order: completion` speaks ready replies first; set `member` to
retain configured member order. Use `parallel: false` when each member should
see earlier answers from the current round.

Run `mcbeak --doctor` to check the team, audio devices, voice models,
workspaces, CLI installation and permission flags without sending model prompts.
Capture currently requires `audio.sample_rate: 16000`. Timing logs include a
turn ID, transcription completion, backend response, and first playback start.
Pending audio is capped at three utterances and shared history at 100 turns.

Tmux launches honor configured workspaces and permission flags, and use stable
pane IDs. An existing cockpit is refused because its launch permissions cannot
be verified. Team replies use headless adapters, including in tmux mode; tmux
individual requests acknowledge successful delivery rather than reading terminal
output aloud. Local stop sends Ctrl+C to the last addressed pane; new
conversation restarts the configured pane.

Guaardvark's API and CLI do not expose a read-only mode, so a Guaardvark
member set to `permissions: readonly` refuses every request. It runs on your
own machine and is `full` in `config.yaml` and for members added to the team.

## Licence

MIT, see `LICENSE`. The models and libraries it downloads keep their own
licences, listed in `THIRD_PARTY_NOTICES.md`. Not affiliated with or endorsed
by the makers of the assistants it talks to.
