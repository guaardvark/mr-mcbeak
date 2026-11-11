# Mr. McBeak

Hands-free voice front end for several AI coding assistants on one machine:
Claude Code, Codex, Grok, Antigravity, Cursor Agent and a local Guaardvark. Speech is
recognised locally (faster-whisper) and replies are spoken locally (Kokoro or
Piper), each assistant in its own voice.

## Run

```
./setup.sh            # venv, Kokoro and Piper voices, models, ~/.local/bin/mcbeak symlink
mcbeak                # always-on listener (same as mcbeak --listen)
mcbeak --verbose      # same, with the technical log lines in the terminal too
mcbeak --replace      # stop the copy already listening and start this one
mcbeak --stop         # stop the copy that is listening
mcbeak --test-voice   # hear every assistant's voice
mcbeak --cli grok "what is a decorator"        # one text turn, no microphone
mcbeak --cli team "which database should we use"   # ask everyone
```

The terminal shows the conversation: what you said and to whom, each reply,
and a dim line for speech it heard without a name. Only one copy listens at a
time: a second one names the copy already running (PID, start time, terminal)
and exits before it opens the microphone. Ctrl+C or Ctrl+Z quits (Ctrl+Z would
otherwise leave it paused, still holding the microphone). Each session is logged to `~/.local/state/mcbeak/mcbeak.log`, which
includes what was heard and stays on this machine.

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
- Tone of voice (`emotion.enabled`): a local model (emotion2vec+ large, about
  70 ms on the CPU, alongside transcription) listens to how you said it. A clear
  reading goes to the assistant with your words as a hint, such as "sounds
  annoyed or angry", and shows on screen next to what you said; neutral or
  unclear speech adds nothing. On the CREMA-D test set (actors, 1,117 clips) at
  the default threshold, 82% of emotional clips got the right hint and 7% of
  neutral ones got a hint. The hint leaves the machine with the words, which is
  why it is a switch.
- While an assistant works, a soft two-note pulse plays every few seconds
  (`system.thinking_cue`), so a long wait does not sound like a dead microphone.
- Each assistant's CLI session is resumed on every turn, so it remembers the
  conversation.

## Safety

Supported adapters run read-only by default (Claude plan mode, Antigravity plan mode,
Cursor ask mode, Codex's read-only sandbox, Grok with write, edit and shell tools
denied). Set
`permissions: full` on an assistant in `config.yaml` to let voice commands edit
files or run commands. Remember the microphone hears everyone in the room.

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
| Tone of voice | emotion2vec+ large (Ma et al., 2024), ONNX export | emotion2vec/emotion2vec_plus_large; Voldorn/emotion2vec-plus-large-onnx | FunASR Model Open Source License 1.1 |
| Voices | Kokoro v1.0 | hexgrad/Kokoro-82M via kokoro-onnx | Apache-2.0 |
| Voices | Piper LibriTTS, Ryan | rhasspy/piper-voices | see each voice's model card |

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
