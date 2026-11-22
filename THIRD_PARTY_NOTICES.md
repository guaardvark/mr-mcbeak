# Third-party notices

Mr. McBeak's own code is under the MIT licence in `LICENSE`. It downloads or
installs the parts below on your machine. Only the small LibriTTS voice config
(`models/piper/en_US-libritts-high.onnx.json`) is stored in this repository;
each part keeps its own licence.

## Models

| Use | Model | Source | Licence |
|---|---|---|---|
| Speech recognition | Whisper large-v3-turbo, small.en (CTranslate2) | OpenAI; mobiuslabsgmbh and Systran conversions | MIT |
| Speech detection | Silero VAD v6 (bundled in faster-whisper) | snakers4/silero-vad | MIT |
| End of turn | Smart Turn v3.2 | pipecat-ai/smart-turn-v3 | BSD-2-Clause |
| Voices | Kokoro v1.0 | hexgrad/Kokoro-82M via kokoro-onnx | Apache-2.0 |
| Voices | Piper `en_US-libritts-high` | rhasspy/piper-voices | CC BY 4.0, trained on LibriTTS (Zen et al., 2019) |
| Tone of voice (off by default) | emotion2vec+ large (Ma et al., 2024), ONNX export | emotion2vec/emotion2vec_plus_large; Voldorn/emotion2vec-plus-large-onnx | FunASR Model Open Source License 1.1 |

The FunASR licence has terms of its own beyond a standard open licence; read it
before you switch tone of voice on in anything you distribute.

## Python libraries

Installed from PyPI by `setup.sh` into a virtual environment. Most are MIT,
BSD or Apache-2.0. Three are GPL-3.0-or-later: `piper-tts` (1.3 and later),
`phonemizer`, and the eSpeak NG library bundled by `espeakng-loader`. They are
used as installed packages and are not part of this repository; anyone who
redistributes a bundle that includes them takes on the GPL's terms for that
bundle. `certifi` and `tqdm` are MPL-2.0.

## Names

Not affiliated with or endorsed by Anthropic, OpenAI, xAI, Google, Anysphere or
Ollama. Claude, Codex, Grok, Antigravity, Cursor and Ollama are their owners'
trademarks, used here only to name the programs Mr. McBeak talks to.
