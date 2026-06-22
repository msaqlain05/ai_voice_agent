"""
Live Speech-to-Text  ·  faster-whisper  ·  microphone input
─────────────────────────────────────────────────────────────
Listens to your microphone, detects pauses, transcribes with Whisper.

Requirements:
    pip install faster-whisper sounddevice numpy scipy
"""

# ── suppress noisy 3rd-party warnings ─────────────────────────
import logging
import math
import os
import shutil
import textwrap
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
# ──────────────────────────────────────────────────────────────

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

# ──────────────────── CONFIG ────────────────────────────────
SAMPLE_RATE       = 16000   # Whisper expects 16 kHz mono
BLOCK_SIZE        = 4096    # ~0.25 s per callback at 16 kHz
MODEL_SIZE        = "base"  # tiny | base | small | medium | large-v3
DEVICE            = "cpu"   # "cpu" or "cuda"
COMPUTE_TYPE      = "int8"  # int8 (fast CPU) | float16 (GPU)

# ── microphone: set to None for auto-detect, or an int index ──
MIC_DEVICE        = None

# ── devices to skip when auto-detecting ───────────────────────
_SKIP_DEVICE_KEYWORDS = (
    "pulse", "pipewire", "dmix", "surround", "hdmi",
    "bluetooth internal", "internal capture", "monitor of",
    "loopback", "spotify", "sysdefault",
)
# ────────────────────────────────────────────────────────────
SILENCE_THRESHOLD = 0.015   # RMS below this → silence
SILENCE_DURATION  = 1.0     # seconds of silence → end of utterance
MIN_SPEECH_SECS   = 0.5     # discard chunks shorter than this
# ────────────────────────────────────────────────────────────


@dataclass
class RuntimeSettings:
    model_size: str = MODEL_SIZE
    device: str = DEVICE
    compute_type: str = COMPUTE_TYPE
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    cpu_threads: int = 0
    silence_duration: float = SILENCE_DURATION
    silence_threshold: float = SILENCE_THRESHOLD
    min_speech_secs: float = MIN_SPEECH_SECS
    skip_mic_test: bool = False


cfg = RuntimeSettings()


def cuda_available() -> bool:
    try:
        import ctranslate2
        return "cuda" in ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        return False


def apply_fast_preset(language: str | None = None) -> RuntimeSettings:
    """Smallest model + greedy decode + shorter silence wait."""
    use_gpu = cuda_available()
    return RuntimeSettings(
        model_size="tiny.en" if language == "en" else "tiny",
        device="cuda" if use_gpu else "cpu",
        compute_type="float16" if use_gpu else "int8",
        language=language,
        beam_size=1,
        vad_filter=False,
        cpu_threads=os.cpu_count() or 4,
        silence_duration=0.5,
        silence_threshold=0.012,
        min_speech_secs=0.3,
        skip_mic_test=True,
    )
# ────────────────────────────────────────────────────────────

# ANSI
R = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GRN = "\033[92m"; RED = "\033[91m"; CYN = "\033[96m"; YLW = "\033[93m"

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


# ── terminal output (status line vs transcript lines) ─────────
class Terminal:
    """Keep live status on one line; print transcripts on clean new lines."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status_active = False

    def _width(self) -> int:
        return shutil.get_terminal_size(fallback=(80, 24)).columns

    def clear_status(self):
        with self._lock:
            if self._status_active:
                print("\r" + " " * self._width() + "\r", end="", flush=True)
                self._status_active = False

    def status(self, line: str):
        """Overwrite the single status line (VU meter, recording, etc.)."""
        with self._lock:
            padded = line.ljust(self._width())[: self._width()]
            print(f"\r{padded}", end="", flush=True)
            self._status_active = True

    def transcript(self, text: str, ts: str | None = None, meta: str = ""):
        """Print a finished transcript without clobbering from the status line."""
        with self._lock:
            if self._status_active:
                print("\r" + " " * self._width() + "\r", end="", flush=True)
                self._status_active = False

            stamp = ts or datetime.now().strftime("%H:%M:%S")
            indent = "  "
            label = f"{DIM}[{stamp}]{R} "
            wrap_at = max(24, self._width() - len(indent) - 8)

            lines = textwrap.wrap(text, width=wrap_at) or [text]
            print(f"{indent}{label}{GRN}{BOLD}{lines[0]}{R}")
            for extra in lines[1:]:
                print(f"{indent}{' ' * 11}{GRN}{BOLD}{extra}{R}")
            if meta:
                print(f"{indent}{DIM}{meta}{R}")
            print(flush=True)

    def note(self, message: str):
        """Short dim message (skipped clip, no speech, etc.)."""
        with self._lock:
            if self._status_active:
                print("\r" + " " * self._width() + "\r", end="", flush=True)
                self._status_active = False
            print(f"  {DIM}{message}{R}", flush=True)


term = Terminal()
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
# native device sample rate / channels — set by main() before stream opens
DEVICE_NATIVE_RATE: int = SAMPLE_RATE
DEVICE_CHANNELS: int = 1


def to_float32_mono(data: np.ndarray) -> np.ndarray:
    """Convert int16 mic data → normalised float32 mono [-1, 1]."""
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype("float32") / 32768.0


def input_channels(dev_idx: int) -> int:
    """Use 1 or 2 channels — enough for stereo headsets."""
    ch = int(sd.query_devices(dev_idx)["max_input_channels"])
    return 1 if ch < 2 else 2


def resample_to_whisper(audio_f32: np.ndarray, orig_rate: int) -> np.ndarray:
    """Resample float32 mono audio from orig_rate → SAMPLE_RATE (16 kHz)."""
    if orig_rate == SAMPLE_RATE:
        return audio_f32
    g = math.gcd(SAMPLE_RATE, orig_rate)
    return resample_poly(audio_f32, SAMPLE_RATE // g, orig_rate // g).astype("float32")


def rms(audio_f32: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio_f32 ** 2)))


def vu_bar(energy: float, width: int = 20) -> str:
    filled = min(int(energy / 0.3 * width), width)
    colour = GRN if energy < 0.05 else YLW if energy < 0.15 else RED
    return f"{colour}{'█' * filled}{'░' * (width - filled)}{R}"
# ─────────────────────────────────────────────────────────────


# ── shared state ─────────────────────────────────────────────
audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
transcribe_queue: queue.Queue[np.ndarray] = queue.Queue()
speech_buffer: list[np.ndarray] = []
is_speaking = False
silence_start = 0.0
peak_energy = 0.0
no_audio_warned = False
shutdown = threading.Event()
# ─────────────────────────────────────────────────────────────


def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    """sounddevice callback — keep work minimal; resample off-thread."""
    if status:
        warn(f"Stream: {status}")
    mono = to_float32_mono(indata)
    try:
        audio_queue.put_nowait(mono)
    except queue.Full:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        audio_queue.put_nowait(mono)


def transcribe_worker(model: WhisperModel):
    """One worker — transcripts print in order, no overlap."""
    while not shutdown.is_set():
        try:
            audio = transcribe_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            segments, info_obj = model.transcribe(
                audio,
                language=cfg.language,
                beam_size=cfg.beam_size,
                best_of=1,
                vad_filter=cfg.vad_filter,
                without_timestamps=True,
                condition_on_previous_text=False,
            )
            parts = [seg.text.strip() for seg in segments if seg.text.strip()]
            if parts:
                text = " ".join(parts)
                meta = (
                    f"lang={info_obj.language}  "
                    f"confidence={info_obj.language_probability:.0%}"
                )
                term.transcript(text, meta=meta)
            else:
                term.note("[no speech detected in clip]")
        except Exception as exc:
            term.clear_status()
            err(f"Transcription error: {exc}")
        finally:
            transcribe_queue.task_done()


def processing_loop():
    """Read audio queue, detect speech/silence, queue transcription."""
    global speech_buffer, is_speaking, silence_start, peak_energy, no_audio_warned

    frame_count = 0
    while not shutdown.is_set():
        try:
            raw = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        chunk = resample_to_whisper(raw, DEVICE_NATIVE_RATE)
        frame_count += 1
        energy = rms(chunk)
        peak_energy = max(peak_energy, energy)

        if frame_count == 60 and peak_energy < 0.003 and not no_audio_warned:
            no_audio_warned = True
            term.note(
                "No audio detected — try:  python live_speech_to_text.py --mic 13"
            )

        if frame_count % 3 == 0:
            state = f"{YLW}SPEAKING{R}" if is_speaking else f"{DIM}listening{R}"
            term.status(f"  {state}  {vu_bar(energy)}  {DIM}{energy:.4f}{R}")

        if energy > cfg.silence_threshold:
            if not is_speaking:
                is_speaking = True
                silence_start = 0.0
                term.status(f"  {YLW}{BOLD}Recording…{R}  {vu_bar(energy)}")
            speech_buffer.append(chunk)
        elif is_speaking:
            if silence_start == 0.0:
                silence_start = time.time()
            speech_buffer.append(chunk)

            if time.time() - silence_start >= cfg.silence_duration:
                full = np.concatenate(speech_buffer)
                dur = len(full) / SAMPLE_RATE

                speech_buffer = []
                is_speaking = False
                silence_start = 0.0

                if dur >= cfg.min_speech_secs:
                    term.status(f"  {CYN}Transcribing {dur:.1f}s…{R}")
                    transcribe_queue.put(full.copy())
                else:
                    term.note(f"[clip too short ({dur:.2f}s), skipped]")


def score_input_device(d: dict) -> int:
    """Higher score = better mic candidate for auto-detect."""
    name = d["name"].lower()
    ch = d["max_input_channels"]

    if ch < 1:
        return -999
    if any(k in name for k in _SKIP_DEVICE_KEYWORDS):
        return -999

    score = 0
    if ch > 8:
        score -= 40
    elif ch <= 4:
        score += 15

    for word, pts in (
        ("analog", 12), ("headset", 10), ("usb", 8), ("microphone", 10),
        ("wave pro", 20), ("built-in", 5),
    ):
        if word in name:
            score += pts

    if name.strip() == "default":
        score -= 25

    return score


def pick_input_device() -> int | None:
    """Pick the best real microphone, not virtual BT/loopback devices."""
    devices = sd.query_devices()
    ranked = []
    for i, d in enumerate(devices):
        score = score_input_device(d)
        if score > -999:
            ranked.append((score, i, d))

    if not ranked:
        return None

    ranked.sort(key=lambda x: (-x[0], x[2]["max_input_channels"], x[1]))
    return ranked[0][1]


def probe_mic(dev_idx: int, seconds: float = 2.0) -> float:
    """Brief listen — returns peak RMS so we can warn if the mic is silent."""
    native_rate = int(sd.query_devices(dev_idx)["default_samplerate"])
    channels = input_channels(dev_idx)
    blocksize = max(256, int(BLOCK_SIZE * native_rate / SAMPLE_RATE))
    peaks: list[float] = []

    def _cb(indata, frames, time_info, status):
        if status:
            warn(f"Mic test: {status}")
        mono = to_float32_mono(indata)
        chunk = resample_to_whisper(mono, native_rate)
        peaks.append(rms(chunk))

    with sd.InputStream(
        device=dev_idx,
        samplerate=native_rate,
        channels=channels,
        dtype="int16",
        blocksize=blocksize,
        callback=_cb,
    ):
        term.status(f"  {YLW}Mic test — speak now ({seconds:.0f}s)…{R}")
        time.sleep(seconds)

    term.clear_status()
    return max(peaks) if peaks else 0.0


def resolve_input_device(requested: int | None) -> int:
    all_devs = sd.query_devices()
    input_devs = [(i, d) for i, d in enumerate(all_devs) if d["max_input_channels"] > 0]
    if not input_devs:
        err("No input (microphone) devices found!")
        sys.exit(1)

    dev_idx = requested
    if dev_idx is None:
        dev_idx = pick_input_device()
        if dev_idx is None:
            dev_idx = sd.default.device[0]

    if dev_idx < 0 or dev_idx >= len(all_devs):
        err(f"Mic device [{dev_idx}] is invalid.")
        sys.exit(1)
    if all_devs[dev_idx]["max_input_channels"] < 1:
        err(f"Device [{dev_idx}] has no microphone input.")
        sys.exit(1)

    print(f"\n  {BOLD}Available input devices:{R}")
    for i, d in input_devs:
        marker = f"  {GRN}← using{R}" if i == dev_idx else ""
        print(f"    [{i:2d}]  {d['name']}  "
              f"{DIM}(ch={d['max_input_channels']}, "
              f"rate={int(d['default_samplerate'])}){R}{marker}")
    print()

    return dev_idx


def parse_args():
    p = argparse.ArgumentParser(description="Live speech-to-text with faster-whisper")
    p.add_argument(
        "--fast", action="store_true",
        help="Speed mode: tiny model, beam=1, 0.5s silence, skip mic test",
    )
    p.add_argument(
        "--model", default=None,
        help="Whisper model: tiny, tiny.en, base, small, …",
    )
    p.add_argument(
        "--lang", default=None,
        help="Language code (e.g. en) — skips auto-detect, faster",
    )
    p.add_argument(
        "--mic", type=int, default=None,
        help="Microphone device index (see list printed at startup)",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="List input devices and exit",
    )
    p.add_argument(
        "--skip-mic-test", action="store_true",
        help="Skip the quick mic level check at startup",
    )
    return p.parse_args()


def main():
    global cfg
    args = parse_args()

    if args.fast:
        cfg = apply_fast_preset(language=args.lang)
    else:
        cfg = RuntimeSettings(
            model_size=args.model or MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            language=args.lang,
            skip_mic_test=args.skip_mic_test,
        )
        if args.skip_mic_test:
            cfg.skip_mic_test = True

    if args.list_devices:
        print(f"\n{BOLD}Input devices:{R}\n")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  [{i:2d}]  {d['name']}  "
                      f"(ch={d['max_input_channels']}, rate={int(d['default_samplerate'])})")
        print()
        return

    mic_choice = args.mic if args.mic is not None else MIC_DEVICE
    mode = f"{YLW}FAST{R}" if args.fast else "normal"
    print(f"\n{BOLD}{CYN}{'━'*52}{R}")
    print(f"{BOLD}{CYN}   Live Speech-to-Text  (faster-whisper){R}")
    print(f"{BOLD}{CYN}{'━'*52}{R}\n")
    print(f"  Mode        : {mode}")
    print(f"  Model       : {YLW}{cfg.model_size}{R}")
    print(f"  Device      : {YLW}{cfg.device}{R}  |  Compute: {YLW}{cfg.compute_type}{R}")
    print(f"  Beam size   : {YLW}{cfg.beam_size}{R}")
    print(f"  Language    : {YLW}{cfg.language or 'auto'}{R}")
    print(f"  Sample rate : {YLW}{SAMPLE_RATE} Hz{R}")
    print(f"  Silence gap : {YLW}{cfg.silence_duration}s{R}  |  Threshold: {YLW}{cfg.silence_threshold}{R}\n")

    info("Scanning audio devices…")
    dev_idx = resolve_input_device(mic_choice)
    chosen = sd.query_devices(dev_idx)
    ok(f"Using mic [{dev_idx}]: {GRN}{BOLD}{chosen['name']}{R} "
       f"(ch={chosen['max_input_channels']}, "
       f"rate={int(chosen['default_samplerate'])} Hz)")

    if not cfg.skip_mic_test:
        peak = probe_mic(dev_idx)
        if peak < 0.003:
            warn(f"Mic test peak={peak:.4f} — very quiet or wrong device.")
            warn("Try:  --mic 12  (system default)  or  --mic 0  (laptop mic)")
            warn("For BT earbuds, switch to headset/HFP profile in sound settings.")
        else:
            ok(f"Mic test OK (peak level {peak:.4f})")

    info(f"Loading Whisper '{cfg.model_size}' model (downloads once on first run)…")
    model = None
    load_err = None

    def _load():
        nonlocal model, load_err
        try:
            kwargs = dict(
                device=cfg.device,
                compute_type=cfg.compute_type,
            )
            if cfg.cpu_threads:
                kwargs["cpu_threads"] = cfg.cpu_threads
            model = WhisperModel(cfg.model_size, **kwargs)
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

    threading.Thread(target=transcribe_worker, args=(model,), daemon=True).start()
    proc = threading.Thread(target=processing_loop, daemon=True)
    proc.start()

    global DEVICE_NATIVE_RATE, DEVICE_CHANNELS
    native_rate = int(sd.query_devices(dev_idx)["default_samplerate"])
    DEVICE_NATIVE_RATE = native_rate
    DEVICE_CHANNELS = input_channels(dev_idx)
    native_blocksize = int(BLOCK_SIZE * native_rate / SAMPLE_RATE)

    info(f"Opening microphone @ {native_rate} Hz, {DEVICE_CHANNELS} ch…")
    try:
        stream = sd.InputStream(
            device=dev_idx,
            samplerate=native_rate,
            channels=DEVICE_CHANNELS,
            dtype="int16",
            blocksize=native_blocksize,
            callback=audio_callback,
        )
        stream.start()
        ok(f"Stream open: {native_rate} Hz native → {SAMPLE_RATE} Hz Whisper\n")
    except Exception as e:
        err(f"Failed to open mic stream: {e}")
        info("Retrying with system default device…")
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

    print(f"  {GRN}{BOLD}Speak into your microphone — pause briefly after each phrase.{R}")
    print(f"  {DIM}Transcripts appear below. Press Ctrl+C to quit.{R}")
    print(f"  {'─'*48}\n")

    try:
        while not shutdown.is_set():
            time.sleep(0.3)
            if not proc.is_alive():
                err("Processing thread died — exiting.")
                break
    except KeyboardInterrupt:
        shutdown.set()
        term.clear_status()
        print(f"\n  {YLW}Stopped. Goodbye!{R}\n")
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
