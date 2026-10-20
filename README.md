# Jarvis Voice Cockpit

Hands-free voice front end for several AI coding assistants on one machine:
Claude Code, Grok, Antigravity, Cursor Agent and a local Guaardvark. Speech is
recognised locally (faster-whisper) and replies are spoken locally (Piper), each
assistant in its own voice.

## Run

```
./setup.sh            # venv, Piper voices, ~/.local/bin/jarvis symlink
jarvis --listen       # always-on listener
jarvis --test-voice   # hear every assistant's voice
jarvis --cli grok "what is a decorator"        # one text turn, no microphone
jarvis --cli team "which database should we use"   # ask everyone
```

## Talking to it

- `Hey Claude ...`, `Hey Grok ...`, `Hey Antigravity ...`, `Hey Cursor ...`,
  `Hey Guaardvark ...` address one assistant. The wake name must come within the
  first three words; a name mentioned later in a sentence does not re-route.
- `Hey Jarvis ...` goes to the last assistant you spoke to, or the default.
- `Hey Team ...` asks every member of the council. They answer one after another
  in their own voices, and each one is shown what the others just said.
- For 30 seconds after a reply you can keep talking without a wake word.
- Each assistant's CLI session is resumed on every turn, so it remembers the
  conversation. Speech heard while an assistant was busy is dropped after a few
  seconds rather than replayed late.

## Safety

Every adapter runs read-only by default (Claude plan mode, Antigravity plan mode,
Cursor ask mode, Grok with write, edit and shell tools denied). Set
`permissions: full` on an assistant in `config.yaml` to let voice commands edit
files or run commands. Remember the microphone hears everyone in the room.

## Config

`config.yaml` holds the wake words, voices and the per-assistant keys
`permissions`, `model`, `workspace` and `timeout_seconds`. The `council` section
lists who answers to "Hey Team". STT model and device are under `stt`; on a
machine with a free GPU, `small.en` with `device: cuda` and `compute_type:
float16` is the first thing to try.

## Tests

```
.venv/bin/python -m unittest discover tests
```
