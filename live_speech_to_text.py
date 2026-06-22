"""
Live Speech-to-Text  ·  faster-whisper  ·  microphone input
─────────────────────────────────────────────────────────────
Listens to your microphone, detects pauses, transcribes with Whisper.

Requirements:
    pip install faster-whisper sounddevice numpy scipy
"""

# ── suppress noisy 3rd-party warnings ─────────────────────────
import logging
import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
# ──────────────────────────────────────────────────────────────

import queue
import sys
import threading
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ──────────────────── CONFIG ────────────────────────────────
SAMPLE_RATE       = 16000   # Whisper expects 16 kHz mono
BLOCK_SIZE        = 4096    # ~0.25 s per callback at 16 kHz
MODEL_SIZE        = "base"  # tiny | base | small | medium | large-v3
DEVICE            = "cpu"   # "cpu" or "cuda"
COMPUTE_TYPE      = "int8"  # int8 (fast CPU) | float16 (GPU)

# ── microphone: set to None for auto-detect, or an int index ──
MIC_DEVICE        = 13      # 13 = "Wave Pro" (your earbuds)

# ── silence detection (on normalised float32 scale 0.0 – 1.0) ──
SILENCE_THRESHOLD = 0.015   # RMS below this → silence
SILENCE_DURATION  = 1.0     # seconds of silence → end of utterance
MIN_SPEECH_SECS   = 0.5     # discard chunks shorter than this
# ────────────────────────────────────────────────────────────

# ANSI
R = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GRN = "\033[92m"; RED = "\033[91m"; CYN = "\033[96m"; YLW = "\033[93m"; MAG = "\033[95m"

# ── logging ──────────────────────────────────────────────────
_log = logging.getLogger("stt")
_log.setLevel(logging.DEBUG)
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter(f"{DIM}%(asctime)s{R}  %(message)s", "%H:%M:%S"))
_log.addHandler(_h)

def info(m): _log.info(f"{CYN}{m}{R}")
def ok(m):   _log.info(f"{GRN}✓ {m}{R}")
def warn(m): _log.warning(f"{YLW}⚠  {m}{R}")
def err(m):  _log.error(f"{RED}✗  {m}{R}")
# ─────────────────────────────────────────────────────────────


# ── spinner ──────────────────────────────────────────────────
class Spinner:
    _f = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._t    = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        i = 0
        while not self._stop.is_set():
            print(f"\r  {CYN}{self._f[i%10]}{R}  {self.label}  ", end="", flush=True)
            i += 1; time.sleep(0.1)
    def __enter__(self):  self._t.start(); return self
    def __exit__(self, *_): self._stop.set(); self._t.join(); print("\r" + " "*60 + "\r", end="", flush=True)
# ─────────────────────────────────────────────────────────────


# ── audio utils ──────────────────────────────────────────────
# native device sample rate — set by main() before stream opens
DEVICE_NATIVE_RATE: int = SAMPLE_RATE


def to_float32_mono(data: np.ndarray) -> np.ndarray:
    """
    Convert raw int16 microphone data → normalised float32 mono [-1, 1].
    data shape: (frames, channels) or (frames,)
    """
    if data.ndim > 1:
        data = data[:, 0]           # keep only the first channel → mono
    return data.astype("float32") / 32768.0


def resample_to_whisper(audio_f32: np.ndarray, orig_rate: int) -> np.ndarray:
    """Resample float32 mono audio from orig_rate → SAMPLE_RATE (16 kHz)."""
    if orig_rate == SAMPLE_RATE:
        return audio_f32
    from scipy.signal import resample_poly
    import math
    g = math.gcd(SAMPLE_RATE, orig_rate)
    return resample_poly(audio_f32, SAMPLE_RATE // g, orig_rate // g).astype("float32")


def rms(audio_f32: np.ndarray) -> float:
    """RMS energy of a normalised float32 array."""
    return float(np.sqrt(np.mean(audio_f32 ** 2)))


def vu_bar(energy: float, width: int = 20) -> str:
    """Mini ASCII VU meter for the terminal."""
    filled = min(int(energy / 0.3 * width), width)
    colour = GRN if energy < 0.05 else YLW if energy < 0.15 else RED
    return f"{colour}{'█' * filled}{'░' * (width - filled)}{R}"
# ─────────────────────────────────────────────────────────────


# ── shared state ─────────────────────────────────────────────
audio_queue:   queue.Queue = queue.Queue()
speech_buffer: list[np.ndarray] = []   # float32 mono chunks
is_speaking    = False
silence_start  = 0.0
# ─────────────────────────────────────────────────────────────


def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    """sounddevice callback — runs in a dedicated OS thread."""
    if status:
        warn(f"Stream: {status}")
    # convert to float32 mono, then resample to 16 kHz for Whisper
    mono = to_float32_mono(indata)
    resampled = resample_to_whisper(mono, DEVICE_NATIVE_RATE)
    audio_queue.put(resampled)


def transcribe_chunk(model: WhisperModel, audio: np.ndarray):
    """Run Whisper on a float32 mono chunk and print the transcript."""
    try:
        segments, info_obj = model.transcribe(
            audio,
            language=None,          # auto-detect
            beam_size=5,
            vad_filter=True,        # Whisper's built-in VAD
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )
        parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        if parts:
            text = " ".join(parts)
            ts   = datetime.now().strftime("%H:%M:%S")
            print(
                f"\n  {DIM}[{ts}]{R} {GRN}{BOLD}{text}{R}  "
                f"{DIM}(lang={info_obj.language} "
                f"p={info_obj.language_probability:.2f}){R}\n",
                flush=True,
            )
        else:
            # model ran but produced no text — show a dim note
            print(f"\r  {DIM}[no speech detected]{R}          ", end="", flush=True)
    except Exception as exc:
        err(f"Transcription error: {exc}")


def processing_loop(model: WhisperModel):
    """Read audio queue, detect speech/silence, fire transcription."""
    global speech_buffer, is_speaking, silence_start

    frame_count = 0
    while True:
        try:
            chunk = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        frame_count += 1
        energy = rms(chunk)

        # ── show live VU meter every N frames ──
        if frame_count % 3 == 0:
            state  = f"{YLW}🎙 SPEAK{R}" if is_speaking else f"{DIM}  idle {R}"
            bar    = vu_bar(energy)
            lvl    = f"{energy:.4f}"
            print(f"\r  {state}  {bar}  {DIM}{lvl}{R}   ", end="", flush=True)

        if energy > SILENCE_THRESHOLD:
            # ── speech ──────────────────────────────
            if not is_speaking:
                is_speaking   = True
                silence_start = 0.0
                print(f"\r  {YLW}{BOLD}🎙  Recording speech…{R}             ", flush=True)
            speech_buffer.append(chunk)

        else:
            # ── silence ─────────────────────────────
            if is_speaking:
                if silence_start == 0.0:
                    silence_start = time.time()
                speech_buffer.append(chunk)   # include trailing silence

                elapsed = time.time() - silence_start
                if elapsed >= SILENCE_DURATION:
                    # ── utterance complete ───────────
                    full = np.concatenate(speech_buffer)
                    dur  = len(full) / SAMPLE_RATE

                    speech_buffer = []
                    is_speaking   = False
                    silence_start = 0.0

                    if dur >= MIN_SPEECH_SECS:
                        print(
                            f"\r  {CYN}⚙  Transcribing {dur:.1f}s of audio…{R}     ",
                            flush=True,
                        )
                        threading.Thread(
                            target=transcribe_chunk,
                            args=(model, full),
                            daemon=True,
                        ).start()
                    else:
                        print(
                            f"\r  {DIM}[clip too short ({dur:.2f}s), skipped]{R}   ",
                            end="", flush=True,
                        )


def pick_input_device() -> int | None:
    """
    Return the best input device index.
    Prefer a real hardware mic over "default" or virtual aggregates.
    """
    devices = sd.query_devices()
    candidates = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] < 1:
            continue
        name = d["name"].lower()
        # skip virtual/aggregate devices that often have silly channel counts
        if any(k in name for k in ("pulse", "pipewire", "dmix", "surround", "hdmi")):
            continue
        candidates.append((i, d))

    # prefer devices with a small channel count (real mics have 1–2 ch)
    candidates.sort(key=lambda x: x[1]["max_input_channels"])

    if candidates:
        idx, d = candidates[0]
        return idx
    return None   # fall back to sounddevice default


def main():
    print(f"\n{BOLD}{CYN}{'━'*52}{R}")
    print(f"{BOLD}{CYN}   🎤  Live Speech-to-Text  (faster-whisper){R}")
    print(f"{BOLD}{CYN}{'━'*52}{R}\n")
    print(f"  Model       : {YLW}{MODEL_SIZE}{R}")
    print(f"  Device      : {YLW}{DEVICE}{R}  |  Compute: {YLW}{COMPUTE_TYPE}{R}")
    print(f"  Sample rate : {YLW}{SAMPLE_RATE} Hz{R}")
    print(f"  Silence gap : {YLW}{SILENCE_DURATION}s{R}  |  Threshold: {YLW}{SILENCE_THRESHOLD}{R}\n")

    # ── 1. pick microphone ────────────────────────────────────
    info("Scanning audio devices…")
    all_devs = sd.query_devices()
    input_devs = [(i, d) for i, d in enumerate(all_devs) if d["max_input_channels"] > 0]
    if not input_devs:
        err("No input (microphone) devices found!")
        sys.exit(1)

    print(f"\n  {BOLD}Available input devices:{R}")
    for i, d in input_devs:
        marker = f"  {GRN}← SELECTED{R}" if i == MIC_DEVICE else ""
        print(f"    [{i:2d}]  {d['name']}  "
              f"{DIM}(ch={d['max_input_channels']}, "
              f"rate={int(d['default_samplerate'])}){R}{marker}")
    print()

    # use configured device or auto-pick
    if MIC_DEVICE is not None:
        dev_idx = MIC_DEVICE
    else:
        dev_idx = pick_input_device()
        if dev_idx is None:
            dev_idx = sd.default.device[0]

    chosen = sd.query_devices(dev_idx)
    ok(f"Using mic [{dev_idx}]: {GRN}{BOLD}{chosen['name']}{R} "
       f"(ch={chosen['max_input_channels']}, "
       f"rate={int(chosen['default_samplerate'])} Hz)")

    # ── 2. load Whisper model ─────────────────────────────────
    info(f"Loading Whisper '{MODEL_SIZE}' model (downloads once ~150 MB)…")
    model = None
    load_err = None

    def _load():
        nonlocal model, load_err
        try:
            model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        except Exception as e:
            load_err = e

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    with Spinner("Loading model…"):
        t.join()

    if load_err:
        err(f"Model load failed: {load_err}")
        sys.exit(1)

    ok("Model ready!\n")

    # ── 3. start processing thread ────────────────────────────
    proc = threading.Thread(target=processing_loop, args=(model,), daemon=True)
    proc.start()

    # ── 4. open microphone stream ─────────────────────────────
    # use the device's native sample rate to avoid PortAudio resampling errors
    global DEVICE_NATIVE_RATE
    native_rate = int(sd.query_devices(dev_idx)["default_samplerate"])
    DEVICE_NATIVE_RATE = native_rate

    # scale blocksize proportionally so each callback is still ~0.25 s
    native_blocksize = int(BLOCK_SIZE * native_rate / SAMPLE_RATE)

    info(f"Opening microphone stream @ {native_rate} Hz → resampling to {SAMPLE_RATE} Hz…")
    try:
        stream = sd.InputStream(
            device=dev_idx,
            samplerate=native_rate,
            channels=1,
            dtype="int16",
            blocksize=native_blocksize,
            callback=audio_callback,
        )
        stream.start()
        ok(f"Stream open: {native_rate} Hz native → {SAMPLE_RATE} Hz Whisper, "
           f"blocksize={native_blocksize}\n")
    except Exception as e:
        err(f"Failed to open mic stream: {e}")
        # fallback: try pipewire/pulseaudio default which does software resampling
        info("Retrying with system default device (pipewire)…")
        try:
            DEVICE_NATIVE_RATE = SAMPLE_RATE
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=BLOCK_SIZE,
                callback=audio_callback,
            )
            stream.start()
            ok(f"Fallback stream open at {SAMPLE_RATE} Hz\n")
        except Exception as e2:
            err(f"Fallback also failed: {e2}")
            sys.exit(1)

    print(f"  {GRN}{BOLD}Speak into your microphone — Ctrl+C to quit{R}")
    print(f"  {'─'*48}\n")

    try:
        while True:
            time.sleep(0.3)
            if not proc.is_alive():
                warn("Processing thread died — restarting…")
                proc = threading.Thread(target=processing_loop, args=(model,), daemon=True)
                proc.start()
    except KeyboardInterrupt:
        print(f"\n\n  {YLW}Stopped. Goodbye! 👋{R}\n")
    finally:
        try:
            stream.stop(); stream.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
