# Mr. McBeak

Hands-free voice front end for several AI coding assistants on one machine:
Claude Code, Grok, Antigravity, Cursor Agent and a local Guaardvark. Speech is
recognised locally (faster-whisper) and replies are spoken locally (Piper), each
assistant in its own voice.

## Run

```
./setup.sh            # venv, Piper voices, ~/.local/bin/mcbeak symlink
mcbeak --listen       # always-on listener
mcbeak --test-voice   # hear every assistant's voice
mcbeak --cli grok "what is a decorator"        # one text turn, no microphone
mcbeak --cli team "which database should we use"   # ask everyone
```

## Talking to it

- `Hey Claude ...`, `Hey Grok ...`, `Hey Antigravity ...`, `Hey Cursor ...`,
  `Hey Guaardvark ...` address one assistant. The wake name must come within the
  first three words; a name mentioned later in a sentence does not re-route.
- `Mr. McBeak ...` goes to the last assistant you spoke to, or the default.
  The names in `config.yaml` are passed to Whisper as hint words, so rare names
  like these are heard reliably.
- `Hey Team ...` asks every member of the council. They answer one after another
  in their own voices. Parallel replies share prior context; sequential replies
  also see earlier answers from the current round.
- For 30 seconds after a reply you can keep talking without a wake word.
- Anything you add while an assistant is still working on your question joins
  that question, so it answers both together instead of losing the first.
- While an assistant works, a soft two-note pulse plays every few seconds
  (`system.thinking_cue`), so a long wait does not sound like a dead microphone.
- Each assistant's CLI session is resumed on every turn, so it remembers the
  conversation.

## Safety

Supported adapters run read-only by default (Claude plan mode, Antigravity plan mode,
Cursor ask mode, Grok with write, edit and shell tools denied). Set
`permissions: full` on an assistant in `config.yaml` to let voice commands edit
files or run commands. Remember the microphone hears everyone in the room.

## Telemetry

Speech recognition (faster-whisper) and the voices (Piper) run on this machine.
What leaves it is what you say to the assistant you name, sent by that
assistant's own CLI. Everything else is switched off in `core/privacy.py`, which
Mr. McBeak applies to itself before any library loads and to every CLI it starts:
ONNX Runtime usage events, Hugging Face pings (the Whisper model is read from
the local cache; the network is used once, to download it on first run), and
the CLIs' own usage metrics, crash reports, Grok's analytics and session-trace
upload, and Claude Code's update checks. Antigravity's usage telemetry is an
account setting, and Cursor's is governed by its Privacy Mode; neither has a
switch Mr. McBeak can set.

## Config

`config.yaml` holds the wake words, voices and the per-assistant keys
`permissions`, `model`, `workspace` and `timeout_seconds`. Claude also takes
`effort` (low to max) and `mcp`: `none` by default for voice turns, `all`, or a
list of your MCP server names. Loading MCP servers added about 1.5 s and $0.05 to
every short turn when measured, so add only the ones you want by voice. The `council` section
lists who answers to "Hey Team". STT model and device are under `stt`; on a
machine with a free GPU, `small.en` with `device: cuda` and `compute_type:
float16` is the first thing to try.

## Tests

```
.venv/bin/python -m unittest discover tests
```

## Reliability and controls

The follow-up window starts after the reply finishes, including interrupted
replies. During an active conversation, say `stop`, `go to sleep`,
`new conversation`, or `repeat that`. While passive, include a wake word.
New conversation resets the active assistant (all members for Team) and clears
shared history. Repeat replays the last spoken reply without another model call.

Interruptions cancel the current turn and suppress remaining council replies.
CLI process groups are terminated; HTTP calls already in progress finish within
the configured timeout, and their cancelled responses are discarded. Backend
turns are serialized while transcription remains available for local controls.
Headphones are still advised: energy-based barge-in cannot distinguish speaker
output from your voice as reliably as acoustic echo cancellation.

Council requests run in parallel by default, using a snapshot of prior context.
`council.reply_order: completion` speaks ready replies first; set `member` to
retain configured member order. Use `parallel: false` when each member should
see earlier answers from the current round.

Run `mcbeak --doctor` to check configuration, audio devices, voice models,
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

Guaardvark's API and CLI do not currently expose verified read-only enforcement.
Its default read-only adapter refuses execution and is omitted from tmux panes.
Only explicitly configured `permissions: full` permits Guaardvark execution.
