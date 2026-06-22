# ai_voice_agent

Live speech-to-text using your microphone and [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python live_speech_to_text.py
```

### Super fast mode (recommended)

```bash
python live_speech_to_text.py --fast --mic 13
```

English only (even faster — skips language detection):

```bash
python live_speech_to_text.py --fast --lang en --mic 13
```

`--fast` enables:
- `tiny` model (smallest, ~4× faster than `base`)
- `beam_size=1` (greedy decode)
- 0.5s silence wait (instead of 1.0s)
- skips Whisper VAD (you already have silence detection)
- skips mic test at startup
- uses GPU automatically if CUDA is available

Use a specific microphone (recommended if auto-detect picks the wrong one):

```bash
python live_speech_to_text.py --mic 13
```

List all input devices:

```bash
python live_speech_to_text.py --list-devices
```

Speak into your mic, pause for about 1 second when you finish a phrase, and the transcript prints in the terminal.

## Config

Edit the top of `live_speech_to_text.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MIC_DEVICE` | `None` | Mic index, or `None` to auto-detect |
| `MODEL_SIZE` | `base` | `tiny` = fastest, `base` = balanced |
| `SILENCE_DURATION` | `1.0` | Lower = text appears sooner (try `0.5`) |
| `DEVICE` / `COMPUTE_TYPE` | `cpu` / `int8` | `cuda` / `float16` if you have NVIDIA GPU |

For GPU: set `DEVICE = "cuda"` and `COMPUTE_TYPE = "float16"`, or just use `--fast`.