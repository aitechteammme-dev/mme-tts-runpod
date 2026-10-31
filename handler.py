"""MME TTS worker for RunPod Serverless (Chatterbox Multilingual).

What changed against the previous handler:
  * The voice is prepared ONCE per worker, at start-up, and kept by NAME (the baked /app/ref_voice.wav as DEFAULT_VOICE, plus any /app/voices/<name>.wav).
    A request picks it with {"input": {"voice": "<name>"}}; no reference audio is read, hashed or prepared per request.
  * Reference audio in the request (reference_audio_base64) is IGNORED unless ALLOW_REQUEST_REF=1. Old clients that still send it keep working
    (the baked voice is used) and simply stop paying for it on the worker; the client should stop sending it to save ~0.77 MB of upload per request.
  * New output_format "ulaw8k": 8 kHz G.711 mu-law, 8000 bytes per second of speech, ready for Plivo. "wav" (default) and "pcm16" are unchanged.
  * Start-up logs the voice name and how long preparing it took.
  * `python handler.py --self-test` checks everything that does not need a GPU (no torch, no chatterbox, no network, no model).

Request:  {"input": {"text": "...", "language": "hi", "voice": "utkarsh", "output_format": "ulaw8k",
                     "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8}}      (voice / output_format / knobs optional)
Response: {"audio_base64", "format", "sample_rate", "language", "voice", "timing": {...}}  or  {"error": "..."}
"""
import base64
import contextlib
import copy
import glob
import inspect
import io
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import warnings

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Both are read by libraries when THEY are first imported (huggingface_hub, spacy_pkuseg), so they are set here, before anything imports them. An explicit setting
# from the environment is kept and then rejected by pinned.runtime.verify_tokenizer_assets if it would allow a download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PKUSEG_HOME", "/app/pkuseg")

from pinned import pins
from pinned import runtime as pinned_runtime

# The verified checkpoint directory (baked by pinned/fetch_checkpoint.py at image build time).
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
# Reference voice baked into the image. Override with env vars.
DEFAULT_REF_WAV = os.getenv("DEFAULT_REF_WAV", "/app/ref_voice.wav")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "utkarsh")          # the name of the voice made from DEFAULT_REF_WAV
VOICES_DIR = os.getenv("VOICES_DIR", "/app/voices")            # optional: more voices, one <name>.wav each
# Legacy: accept reference_audio_base64 in a request (0 = ignored). When on, that voice is prepared on EVERY such request (slow).
ALLOW_REQUEST_REF = os.getenv("ALLOW_REQUEST_REF", "0") == "1"
WARMUP_TEXT = "नमस्ते, कैसे हैं आप?"
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "600"))
MAX_REF_BYTES = 15 * 1024 * 1024
OUTPUT_FORMATS = ("wav", "pcm16", "ulaw8k")

ALLOWED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr",
    "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
    "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}

# ---------------------------------------------------------------------------------------------------------- speaking pace (language-specific)
# Chatterbox has no speaking-rate control (the `exaggeration` / `cfg_weight` / `temperature` knobs change delivery, not a rate), so the pace is set AFTER synthesis by a pitch-preserving time-stretch of the
# finished waveform (WSOLA: overlap-add of waveform segments aligned by cross-correlation; the sample rate and the voice's pitch are untouched; no sample-rate resampling). Speed 1.0 = the audio is returned untouched
# (the same array), which is what Hindi gets by default. speed < 1 slows the speech down (0.85 = 17.6 % longer). Configure per language with the environment variables below; an invalid value is ignored (logged) and 1.0 is used.
PACE_MIN, PACE_MAX = 0.5, 1.5
PACE_DEFAULTS = {"en": 0.85, "hi": 1.0, "other": 1.0}
PACE_ENV = {"en": "TTS_SPEED_EN", "hi": "TTS_SPEED_HI", "other": "TTS_SPEED_OTHER"}
_ENGLISH_NAMES = {"en", "eng", "english"}
_HINDI_NAMES = {"hi", "hin", "hindi", "hinglish"}      # Hinglish text is synthesized as Hindi (the client tags it `hi`): it is never slowed down as English
STRETCH_FRAME_S, STRETCH_TOL_S = 0.030, 0.010          # WSOLA segment length and alignment search radius


def pace_language(language) -> str:
    """The pace class of a language tag: 'en' for English however it is written (en, EN, english, en-US, en_GB), 'hi' for Hindi/Hinglish, else the lower-case primary subtag ('' when missing)."""
    s = str(language or "").strip().lower().replace("_", "-")
    primary = s.split("-")[0]
    if primary in _ENGLISH_NAMES:
        return "en"
    if primary in _HINDI_NAMES:
        return "hi"
    return primary


def parse_speed(raw, default: float, name: str = "speed", warn=None) -> float:
    """A speed from configuration. Unset/empty -> `default`. Not a number, not finite or outside [PACE_MIN, PACE_MAX] -> 1.0 (no processing) and a warning: a bad setting never crashes the worker."""
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        v = float(str(raw).strip())
    except ValueError:
        v = float("nan")
    if not (np.isfinite(v) and PACE_MIN <= v <= PACE_MAX):
        if warn:
            warn(f"[TTS] WARNING: {name}={raw!r} is not a number in [{PACE_MIN}, {PACE_MAX}]: using 1.0 (no time-stretch)")
        return 1.0
    return v


def load_pace_config(env, warn=None) -> dict:
    return {k: parse_speed(env.get(PACE_ENV[k]), PACE_DEFAULTS[k], PACE_ENV[k], warn) for k in PACE_DEFAULTS}


def pace_speed(language, pace=None) -> float:
    """The configured speed for a language tag. English variants share one setting; Hindi/Hinglish have their own; missing or unknown languages use the 'other' setting (1.0 by default)."""
    pace = PACE if pace is None else pace
    k = pace_language(language)
    return pace["en"] if k == "en" else pace["hi"] if k == "hi" else pace["other"]


def time_stretch(audio, sr: int, speed: float):
    """Pitch-preserving time-stretch by WSOLA. Returns float32 audio of about len(audio) / speed samples at the SAME sample rate. Returned UNCHANGED (same array) for speed 1.0 or an invalid speed, empty audio, and clips
    shorter than three segments (90 ms: too short to stretch cleanly). All-zero audio gives silence of the stretched length. Non-finite samples are replaced by 0. Pure numpy, no JIT, no new dependency."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0 or not np.isfinite(speed) or speed <= 0 or abs(speed - 1.0) < 1e-3:
        return audio
    frame = int(round(STRETCH_FRAME_S * sr)) // 2 * 2
    hop, tol = frame // 2, int(round(STRETCH_TOL_S * sr))
    if x.size < 3 * frame:
        return audio
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n_out = int(round(x.size / speed))
    if not x.any():
        return np.zeros(n_out, dtype=np.float32)
    win = np.hanning(frame + 1)[:-1].astype(np.float32)           # periodic Hann: neighbouring segments at 50 % overlap sum to exactly 1
    win0 = win.copy()
    win0[:hop] = 1.0                                              # no fade-in on the very first segment
    xp = np.concatenate([np.zeros(tol, np.float32), x, np.zeros(frame + 2 * tol + hop, np.float32)])    # index i of x is index i + tol of xp
    n_frames = -(-n_out // hop)
    out = np.zeros(n_frames * hop + frame, dtype=np.float32)
    out[:frame] += xp[tol:tol + frame] * win0
    prev = 0
    for k in range(1, n_frames + 1):
        target = int(round(k * hop * speed))
        nat = prev + hop                                          # where the previous segment would naturally continue
        template = xp[nat + tol:nat + tol + frame]
        lo = target - tol
        search = xp[lo + tol:lo + tol + 2 * tol + frame]
        corr = np.correlate(search, template, mode="valid")       # similarity of every candidate start in [target - tol, target + tol]
        cs = np.concatenate([[0.0], np.cumsum(search.astype(np.float64) ** 2)])
        energy = cs[frame:frame + 2 * tol + 1] - cs[:2 * tol + 1]
        p = lo + int(np.argmax(corr / np.sqrt(energy + 1e-9)))
        out[k * hop:k * hop + frame] += xp[p + tol:p + tol + frame] * win
        prev = p
    return np.clip(out[:n_out], -1.0, 1.0).astype(np.float32)


PACE = load_pace_config(os.environ, warn=lambda m: print(m, flush=True))

GEN_LOCK = threading.Lock()
MODEL = None           # the Chatterbox model (a fake one in --self-test)
_torch = None           # set by load_model(); None means "no torch": generate_audio then uses the model's output as it is (self-test)
VOICES = {}             # voice name -> prepared conditionals, built ONCE per worker at start-up
_active_voice = None


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------------------------- model and voices
def loaded_line(seconds: float, t3_label: str) -> str:
    return f"[TTS] model loaded in {seconds:.2f}s (t3_model={t3_label})"


def load_model():
    """Load the model ONCE at start-up (never inside a request). Needs a CUDA GPU. FAIL-CLOSED: the pinned Chatterbox source, the pinned checkpoint files (sha256) and the V3 T3
    file are verified first (pinned/runtime.py); any mismatch raises and the worker does not start. There is no fallback to V2 or to an unpinned download."""
    global MODEL, _torch
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t = time.perf_counter()
    MODEL = pinned_runtime.load_verified_model("cuda", MODEL_DIR, os.environ, log)
    _torch = torch
    log(loaded_line(time.perf_counter() - t, pins.T3_FILE))


def _clone_conds(conds):
    """A copy of a prepared voice that the model may change without touching the stored one. SHALLOW on purpose: Chatterbox's conditionals hold tensors that
    are results of computations (not leaves), and PyTorch refuses to deep-copy those ("Only Tensors created explicitly by the user (graph leaves) support the
    deepcopy protocol"); that error crashed every worker at start-up in the first v3 image. generate() REBINDS `conds.t3` when the exaggeration differs (never
    edits the tensors) and S3Gen converts the entries of `conds.gen` in place: so the object and the small `gen` dict are copied, the tensors are shared."""
    c = copy.copy(conds)
    if isinstance(getattr(c, "gen", None), dict):
        c.gen = dict(c.gen)
    return c


def prepare_voice(name: str, wav_path: str) -> float:
    """Prepare one voice (the slow part of Chatterbox) and keep its conditionals under `name`. Returns the seconds it took."""
    t0 = time.perf_counter()
    MODEL.prepare_conditionals(wav_path)
    conds = getattr(MODEL, "conds", None)
    if conds is None:   # fail at start-up, loudly, instead of serving the wrong voice
        raise RuntimeError("the model has no .conds after prepare_conditionals(): this Chatterbox version cannot cache voices by name")
    VOICES[name] = _clone_conds(conds)
    return time.perf_counter() - t0


def select_voice(name: str) -> None:
    """Make `name` the active voice. A dictionary lookup: no audio is read and nothing is prepared. Call inside GEN_LOCK."""
    global _active_voice
    if name != _active_voice:
        MODEL.conds = _clone_conds(VOICES[name])
        _active_voice = name


def init_voices() -> None:
    """Prepare every voice once and log its name and how long it took."""
    if os.path.exists(DEFAULT_REF_WAV):
        spent = prepare_voice(DEFAULT_VOICE, DEFAULT_REF_WAV)
        log(f"[TTS] voice '{DEFAULT_VOICE}' prepared in {spent:.2f}s (source {DEFAULT_REF_WAV})")
    else:
        log(f"[TTS] WARNING: default reference not found at {DEFAULT_REF_WAV}")
    for path in sorted(glob.glob(os.path.join(VOICES_DIR, "*.wav"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in VOICES:
            log(f"[TTS] WARNING: voice '{name}' in {VOICES_DIR} is skipped: the name is already taken")
            continue
        spent = prepare_voice(name, path)
        log(f"[TTS] voice '{name}' prepared in {spent:.2f}s (source {path})")
    if not VOICES:
        log("[TTS] WARNING: no voice is available: every request will fail until a voice is baked into the image")


def generate_audio(text, language, **gen_kwargs):
    if _torch is None:   # no torch (self-test): the model returns a numpy array
        t_gen = time.perf_counter()
        wav = MODEL.generate(text, language_id=language, **gen_kwargs)
        _P.rec("model_generate_s", time.perf_counter() - t_gen)
        _P.rec("sync_and_host_copy_s", 0.0)
        return np.atleast_1d(np.asarray(wav, dtype=np.float32))
    t_gen = time.perf_counter()
    with _torch.inference_mode():
        wav = MODEL.generate(text, language_id=language, **gen_kwargs)
    _P.rec("model_generate_s", time.perf_counter() - t_gen)
    t_post = time.perf_counter()
    _torch.cuda.synchronize()
    out = np.atleast_1d(wav.detach().float().cpu().squeeze().numpy())
    _P.rec("sync_and_host_copy_s", time.perf_counter() - t_post)       # upstream generate() already returns a CPU tensor (it copies before the watermark), so this is expected to be ~0
    return out


# ---------------------------------------------------------------------------------------------------------- latency profiling (OFF unless V4_PROFILE=1)
# Experimental branch only. With the variable unset every call below is a no-op and the response is byte-for-byte the baseline one. With V4_PROFILE=1 the response gains a `timing_detail` object
# (seconds per stage + GPU / library / worker metadata); the audio is unchanged. The wrappers are installed AFTER the model was verified and warmed up, they call the real functions and return their results
# untouched, and they add a CUDA synchronize at each boundary (so GPU stages are attributed correctly): profile numbers are therefore slightly higher than unprofiled ones. No credentials, audio or text are recorded.
PROFILE = os.getenv("V4_PROFILE", "0") == "1"


class _Profiler:
    def __init__(self):
        self.enabled = False
        self.d = {}
        self.meta = {}

    def begin(self):
        if self.enabled:
            self.d = {}

    def rec(self, key, seconds):
        if self.enabled:
            self.d[key] = self.d.get(key, 0.0) + float(seconds)

    def count(self, key, n):
        if self.enabled:
            self.d[key] = n

    def snapshot(self, total_s, extra):
        if not self.enabled:
            return None
        d = dict(self.d)
        parts = sum(d.get(k, 0.0) for k in ("tokenize_s", "t3_s", "flow_s", "hift_s", "watermark_s"))
        if "model_generate_s" in d:
            d["model_generate_other_s"] = max(0.0, d["model_generate_s"] - parts)
        known = sum(d.get(k, 0.0) for k in ("validate_s", "lock_wait_s", "select_voice_s", "model_generate_s", "sync_and_host_copy_s", "stretch_s", "encode_s"))
        d["handler_unattributed_s"] = max(0.0, total_s - known)
        if d.get("speech_tokens") and d.get("t3_s") is not None:
            d["t3_ms_per_token"] = 1000.0 * d["t3_s"] / d["speech_tokens"]
        out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
        out.update(self.meta)
        out.update(extra)
        return out


_P = _Profiler()


def install_profiling(model, torch_mod=None) -> bool:
    """Wrap the stage functions of the loaded model with timers. Returns False (and changes nothing) if the model does not have the expected attributes."""
    try:
        targets = [(model.tokenizer, "text_to_tokens", "tokenize_s"), (model.t3, "inference", "t3_s"), (model.s3gen, "flow_inference", "flow_s"),
                   (model.s3gen, "hift_inference", "hift_s"), (model.watermarker, "apply_watermark", "watermark_s")]
        for obj, name, _ in targets:
            if not callable(getattr(obj, name)):
                raise AttributeError(name)
    except AttributeError as exc:
        log(f"[TTS_PROFILE] not installed: the model has no {exc}")
        return False
    cuda = torch_mod is not None and torch_mod.cuda.is_available()

    def wrap(fn, key):
        def timed(*args, **kwargs):
            if cuda:
                torch_mod.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*args, **kwargs)
            if cuda:
                torch_mod.cuda.synchronize()
            _P.rec(key, time.perf_counter() - t)
            if key == "t3_s":
                shape = getattr(out, "shape", None)
                if shape is not None and len(shape) == 2:
                    _P.count("speech_tokens", int(shape[1]))
            return out
        return timed
    for obj, name, key in targets:
        setattr(obj, name, wrap(getattr(obj, name), key))      # instance attribute: the class (and the pinned fingerprints) are not touched
    meta = {"profile_mode": True, "python": sys.version.split()[0], "worker_id": os.getenv("RUNPOD_POD_ID", "unavailable"), "sample_rate": getattr(model, "sr", None)}
    if cuda:
        meta.update({"gpu": torch_mod.cuda.get_device_name(0), "torch": torch_mod.__version__, "cuda": torch_mod.version.cuda})
    _P.meta, _P.enabled = meta, True
    log("[TTS_PROFILE] installed: stage timers on tokenizer, T3, S3Gen flow, HiFT vocoder and the watermarker (CUDA-synchronized)")
    return True


def warm_up() -> None:
    """Warm up the CUDA kernels with the default voice so the first real request is fast."""
    if not VOICES:
        return
    with GEN_LOCK:
        select_voice(DEFAULT_VOICE if DEFAULT_VOICE in VOICES else next(iter(VOICES)))
        t = time.perf_counter()
        generate_audio(WARMUP_TEXT, "hi")
        generate_audio(WARMUP_TEXT, "hi")
    log(f"[TTS] warmup done in {time.perf_counter() - t:.2f}s")


# ---------------------------------------------------------------------------------------------------------- audio encoding
def pcm16_to_ulaw(pcm: np.ndarray) -> bytes:
    """G.711 mu-law, the Sun 14-bit reference algorithm: bit-identical to Python's audioop.lin2ulaw (checked on all 65536 int16 values)."""
    seg_end = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
    v = pcm.astype(np.int32) >> 2
    mask = np.where(v < 0, 0x7F, 0xFF)
    v = np.minimum(np.abs(v), 8159) + 33
    seg = (v[:, None] > seg_end[None, :]).sum(axis=1)
    uval = np.where(seg >= 8, 0x7F, (seg << 4) | ((v >> (seg + 1)) & 0x0F))
    return (uval ^ mask).astype(np.uint8).tobytes()


def encode_audio(audio: np.ndarray, sr: int, output_format: str) -> str:
    if output_format == "ulaw8k":   # telephony: 8 kHz mu-law, 8000 bytes per second of speech (the model rate is a multiple of 8 kHz)
        a8 = resample_poly(audio, 1, sr // 8000)
        return base64.b64encode(pcm16_to_ulaw((np.clip(a8, -1.0, 1.0) * 32767.0).astype(np.int16))).decode("ascii")
    if output_format == "pcm16":
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        return base64.b64encode(pcm.tobytes()).decode("ascii")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------------------------------------- the RunPod handler
def handler(job):
    data = job.get("input", {}) or {}

    # Cheap ping so your server can pre-warm without synthesizing anything
    if data.get("warmup"):
        yield {"ok": True}
        return

    # Diagnostic streaming validation action
    if data.get("action") == "validate_streaming":
        try:
            from streaming import gpu_validate_streaming as gv
            out_dir = "/tmp/stream_val"
            os.makedirs(out_dir, exist_ok=True)
            seeds = int(data.get("seeds", 3))
            quick = bool(data.get("quick", True))

            log("[TTS] executing validate_streaming on GPU...")
            t3_res = gv.stage_t3_equiv(MODEL, DEFAULT_REF_WAV, seeds=seeds)
            dec_res = gv.stage_decoder(MODEL, DEFAULT_REF_WAV, out_dir=out_dir, quick=quick)
            lat_res = gv.stage_latency(MODEL, DEFAULT_REF_WAV, runs=1)

            yield {
                "ok": True,
                "action": "validate_streaming",
                "env": gv.env_report(MODEL),
                "t3_equiv": t3_res,
                "decoder": dec_res,
                "latency": lat_res,
            }
        except Exception as exc:
            traceback.print_exc()
            log(f"[TTS_VALIDATE_ERROR] {exc}")
            yield {"error": f"validate_streaming_failed: {exc}"}
        return

    # Streamed audio (checks watermarking policy gate)
    if data.get("stream"):
        allow_unw = os.getenv("ALLOW_UNWATERMARKED_STREAMING", "0") == "1" or data.get("allow_unwatermarked") is True
        if not allow_unw:
            refusal = pinned_runtime.streaming_refusal()
            yield refusal if refusal else {
                "seq": 0,
                "error": "streaming_disabled",
                "detail": "Streamed audio cannot carry Perth watermark. Set ALLOW_UNWATERMARKED_STREAMING=1",
                "final": True
            }
            return

        try:
            from streaming.handler_stream import handler_stream_generator
            yield from handler_stream_generator(job, MODEL, VOICES, DEFAULT_VOICE, gen_lock=GEN_LOCK)
        except Exception as exc:
            traceback.print_exc()
            log(f"[TTS_STREAM_ERROR] {exc}")
            yield {"seq": 0, "error": f"streaming_failed: {type(exc).__name__}: {exc}", "final": True}
        return

    total_start = time.perf_counter()
    _P.begin()

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        yield {"error": "Provide a non-empty input.text"}
        return
    text = text.strip()[:MAX_TEXT_CHARS]

    language = data.get("language", "hi")
    if language not in ALLOWED_LANGUAGES:
        yield {"error": f"Unsupported language: {language}"}
        return

    output_format = data.get("output_format", "wav")  # "wav" (default), "pcm16" (model rate) or "ulaw8k" (8 kHz mu-law)
    if output_format not in OUTPUT_FORMATS:
        yield {"error": "output_format must be 'wav', 'pcm16' or 'ulaw8k'"}
        return
    if output_format == "ulaw8k" and MODEL.sr % 8000:
        yield {"error": f"ulaw8k needs a model rate that is a multiple of 8000 (got {MODEL.sr})"}
        return

    voice = data.get("voice", DEFAULT_VOICE)
    if voice not in VOICES and not (ALLOW_REQUEST_REF and data.get("reference_audio_base64")):
        yield {"error": f"Unknown voice: {voice!r}", "voices": sorted(VOICES)}
        return

    # Optional generation knobs (only passed when provided)
    gen_kwargs = {}
    for key in ("exaggeration", "cfg_weight", "temperature"):
        if key in data:
            gen_kwargs[key] = float(data[key])

    _P.rec("validate_s", time.perf_counter() - total_start)
    try:
        t_before_lock = time.perf_counter()
        with GEN_LOCK:
            _P.rec("lock_wait_s", time.perf_counter() - t_before_lock)
            ref_s = 0.0
            ref_b64 = data.get("reference_audio_base64") if ALLOW_REQUEST_REF else None
            if ref_b64:   # legacy path (off by default): a voice sent in the request is prepared on EVERY such request (slow)
                ref_bytes = base64.b64decode(ref_b64, validate=True)
                if len(ref_bytes) > MAX_REF_BYTES:
                    yield {"error": "Reference audio must be 15 MB or smaller"}
                    return
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    f.write(ref_bytes)
                    f.flush()
                    ref_s = prepare_voice("request", f.name)
                voice = "request"
            t_sel = time.perf_counter()
            select_voice(voice)   # by name: no reference audio is read or prepared per request
            _P.rec("select_voice_s", time.perf_counter() - t_sel)

            t0 = time.perf_counter()
            audio = generate_audio(text, language, **gen_kwargs)
            generate_s = time.perf_counter() - t0

        # Pace (outside GEN_LOCK: CPU work must not hold up the GPU). The synthesized waveform is measured BEFORE it is touched and kept as audio_raw; a failure here falls back to it.
        audio_raw = audio
        raw_audio_s = len(audio_raw) / MODEL.sr
        speed = pace_speed(language)
        stretch_s, raw_peak, raw_rms = 0.0, 0.0, 0.0
        if speed != 1.0 and len(audio_raw):
            raw_peak, raw_rms = float(np.max(np.abs(audio_raw))), float(np.sqrt(np.mean(np.square(audio_raw, dtype=np.float64))))
            t0 = time.perf_counter()
            try:
                audio = time_stretch(audio_raw, MODEL.sr, speed)
            except Exception as exc:                 # noqa: BLE001: pace is cosmetic, the request must still be answered with the original audio
                traceback.print_exc()
                log(f"[TTS_PACE_ERROR] {type(exc).__name__}: {exc}: sending the unstretched audio")
                audio, speed = audio_raw, 1.0
            stretch_s = time.perf_counter() - t0
        _P.rec("stretch_s", stretch_s)

        t0 = time.perf_counter()
        audio_b64 = encode_audio(audio, MODEL.sr, output_format)
        encode_s = time.perf_counter() - t0
        _P.rec("encode_s", encode_s)

        audio_s = len(audio) / MODEL.sr                  # the audio actually returned
        total_s = time.perf_counter() - total_start
        log(
            "[TTS_TIMING] "
            f"chars={len(text)} audio_s={audio_s:.2f} ref_s={ref_s:.3f} voice={voice} "
            f"generate_s={generate_s:.3f} encode_s={encode_s:.3f} "
            f"total_s={total_s:.3f} rtf={generate_s / max(raw_audio_s, 1e-6):.2f} fmt={output_format} "
            f"speed={speed:.2f} raw_audio_s={raw_audio_s:.2f} stretch_s={stretch_s:.3f}"
        )

        response = {
            "audio_base64": audio_b64,
            "format": output_format,
            "sample_rate": 8000 if output_format == "ulaw8k" else MODEL.sr,
            "language": language,
            "voice": voice,
            "timing": {
                "generate_s": round(generate_s, 3),
                "total_s": round(total_s, 3),
                "audio_s": round(audio_s, 3),
            },
        }
        detail = _P.snapshot(total_s, {"chars": len(text), "audio_s": round(audio_s, 3), "output_format": output_format, "response_base64_chars": len(audio_b64),
                                       "speed": speed, "raw_audio_s": round(raw_audio_s, 3), "raw_peak": round(raw_peak, 4), "raw_rms": round(raw_rms, 4)})
        if detail is not None:       # only with V4_PROFILE=1: an extra field, the audio and the baseline fields are unchanged
            response["timing_detail"] = detail
            log("[TTS_PROFILE] " + json.dumps(detail, sort_keys=True))
        yield response

    except Exception as exc:
        traceback.print_exc()
        log(f"[TTS_ERROR] {exc}")
        yield {"error": str(exc), "type": type(exc).__name__}


# ---------------------------------------------------------------------------------------------------------- self-test (no GPU)
def run_self_test() -> int:
    """`python handler.py --self-test`: checks voice caching, the request contract, the three output formats and the mu-law encoder with a FAKE
    model. Needs numpy, scipy and soundfile only: no torch, no chatterbox, no GPU, no network. Exit code 0 = all passed."""
    global MODEL, _torch, VOICES, _active_voice, DEFAULT_REF_WAV, DEFAULT_VOICE, VOICES_DIR, ALLOW_REQUEST_REF, log
    results, lines = [], []

    _real_handler = globals()["handler"]

    def handler(job):
        res = _real_handler(job)
        if inspect.isgenerator(res):
            items = list(res)
            return items[0] if items else {}
        return res

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""), flush=True)

    class GraphConds:
        """Stands in for Chatterbox's Conditionals: its tensors are results of computations (graph tensors) and PyTorch refuses to deep-copy them."""

        def __init__(self, amp):
            self.t3, self.gen = ("t3", amp), {"amp": amp}

        def __deepcopy__(self, memo):
            raise RuntimeError("Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment")

    class FakeModel:
        sr = 24000

        def __init__(self, graph=False):
            self.conds, self.prepares, self.graph = None, [], graph

        def prepare_conditionals(self, path):
            data, _ = sf.read(path, dtype="float32")
            self.prepares.append(os.path.basename(path))
            amp = float(np.abs(data).max())                      # the "voice" of the fake model is its loudness
            self.conds = GraphConds(amp) if self.graph else {"amp": amp}

        def generate(self, text, language_id="hi", **kw):
            c = self.conds
            amp = c["amp"] if isinstance(c, dict) else c.gen["amp"]
            if not isinstance(c, dict):
                c.gen["_touched"] = True                         # S3Gen converts the entries of the reference dict in place
                if "exaggeration" in kw:
                    c.t3 = ("t3-exaggeration", kw["exaggeration"])   # generate() REBINDS conds.t3 when the exaggeration differs
            n = int(self.sr * 0.1 * max(1, len(text.split())))
            t = np.arange(n) / self.sr
            return (amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    def ulaw_ref(v):   # the Sun reference algorithm, one sample, plain Python
        seg_end = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)
        x = v >> 2
        mask = 0x7F if x < 0 else 0xFF
        x = min(-x if x < 0 else x, 8159) + 33
        seg = sum(1 for t in seg_end if x > t)
        return (0x7F if seg >= 8 else ((seg << 4) | ((x >> (seg + 1)) & 0x0F))) ^ mask

    def ulaw_decode(b):   # standard G.711 mu-law decode
        u = ~np.frombuffer(b, dtype=np.uint8).astype(np.int32) & 0xFF
        mag = (((u & 0x0F) << 3) + 0x84 << ((u >> 4) & 7)) - 0x84
        return np.where(u & 0x80, -mag, mag)

    print("self-test (no GPU, fake model)")
    # --- 0. the shipped defaults (read BEFORE the test overrides any setting)
    check("default: a reference sent in the request is ignored (ALLOW_REQUEST_REF is off unless the environment turns it on)",
          ALLOW_REQUEST_REF == (os.getenv("ALLOW_REQUEST_REF", "0") == "1"))
    check("default: the voice is named by DEFAULT_VOICE and read from DEFAULT_REF_WAV (baked into the image)",
          DEFAULT_VOICE == os.getenv("DEFAULT_VOICE", "utkarsh") and DEFAULT_REF_WAV == os.getenv("DEFAULT_REF_WAV", "/app/ref_voice.wav"))
    # --- 1. mu-law encoder
    allv = np.arange(-32768, 32768, dtype=np.int16)
    mine = pcm16_to_ulaw(allv)
    check("mu-law: all 65536 int16 values match the reference algorithm", mine == bytes(ulaw_ref(int(v)) for v in allv))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import audioop
        check("mu-law: identical to audioop.lin2ulaw", mine == audioop.lin2ulaw(allv.tobytes(), 2))
    except ImportError:
        print("  [SKIP] audioop is not available in this Python (removed in 3.13)")
    sr = 24000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
    u = base64.b64decode(encode_audio(tone, sr, "ulaw8k"))
    check("ulaw8k: 1 s of 24 kHz audio is exactly 8000 bytes", len(u) == 8000, f"{len(u)} bytes")
    ref8 = np.clip(resample_poly(tone, 1, 3), -1, 1) * 32767
    dec = ulaw_decode(u)
    if len(dec) == len(ref8):
        snr = 10 * np.log10((ref8 ** 2).mean() / max(((dec - ref8) ** 2).mean(), 1e-9))
        check("ulaw8k: decodes back to the 8 kHz signal (SNR above 25 dB)", snr > 25, f"{snr:.1f} dB")
    else:
        check("ulaw8k: decodes back to the 8 kHz signal (SNR above 25 dB)", False, f"length {len(dec)} != {len(ref8)}")

    # --- 2. voices: prepared once, selected by name
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "voices"))
        sf.write(os.path.join(d, "ref.wav"), 0.2 * np.sin(2 * np.pi * 200 * np.arange(8000) / 16000), 16000)
        sf.write(os.path.join(d, "voices", "priya.wav"), 0.6 * np.sin(2 * np.pi * 200 * np.arange(8000) / 16000), 16000)
        MODEL, _torch, VOICES, _active_voice = FakeModel(), None, {}, None
        DEFAULT_REF_WAV, DEFAULT_VOICE, VOICES_DIR, ALLOW_REQUEST_REF = os.path.join(d, "ref.wav"), "utkarsh", os.path.join(d, "voices"), False
        real_log, log = log, (lambda m: lines.append(m))
        init_voices()
        warm_up()
        log = lambda m: None   # the per-request timing lines are not part of the report   # noqa: E731
        check("start-up logs each voice with its name and preparation time",
              any(l.startswith("[TTS] voice 'utkarsh' prepared in ") and l.split(" prepared in ")[1].split("s")[0].replace(".", "").isdigit() for l in lines)
              and any(l.startswith("[TTS] voice 'priya' prepared in ") for l in lines), "; ".join(lines))
        check("each voice is prepared exactly once at start-up", MODEL.prepares == ["ref.wav", "priya.wav"], str(MODEL.prepares))

        def peak(resp):
            data, _ = sf.read(io.BytesIO(base64.b64decode(resp["audio_base64"])), dtype="float32")
            return float(np.abs(data).max())

        base = {"text": "यह एक छोटा सा परीक्षण वाक्य है", "language": "hi"}
        before = len(MODEL.prepares)
        peaks = {"utkarsh": [], "priya": []}
        for i in range(20):                                        # alternate voices 20 times: no preparation may happen
            v = "priya" if i % 2 else "utkarsh"
            peaks[v].append(peak(handler({"input": {**base, "voice": v}})))
        check("20 requests alternating two voices prepare nothing", len(MODEL.prepares) == before, str(MODEL.prepares))
        check("each request is synthesized with the voice it named",
              all(abs(p - 0.2) < 0.02 for p in peaks["utkarsh"]) and all(abs(p - 0.6) < 0.02 for p in peaks["priya"]), str({k: (min(v), max(v)) for k, v in peaks.items()}))
        r = handler({"input": base})
        check("no voice named = the default voice", r.get("voice") == "utkarsh" and abs(peak(r) - 0.2) < 0.02)
        e = handler({"input": {**base, "voice": "nobody"}})
        check("an unknown voice is an error that lists the voices", "error" in e and e.get("voices") == ["priya", "utkarsh"], str(e))

        # --- 3. a reference sent in the request
        junk = base64.b64encode(b"this is not a wav file").decode()
        r = handler({"input": {**base, "reference_audio_base64": junk}})
        check("reference_audio_base64 is ignored by default (old clients keep working, nothing is decoded or prepared)",
              "error" not in r and r["voice"] == "utkarsh" and len(MODEL.prepares) == before)
        ALLOW_REQUEST_REF = True
        with open(os.path.join(d, "voices", "priya.wav"), "rb") as fh:
            wavb64 = base64.b64encode(fh.read()).decode()
        r1, r2 = handler({"input": {**base, "reference_audio_base64": wavb64}}), handler({"input": {**base, "reference_audio_base64": wavb64}})
        check("ALLOW_REQUEST_REF=1: the legacy path works and prepares on every such request",
              r1.get("voice") == "request" and r2.get("voice") == "request" and len(MODEL.prepares) == before + 2 and abs(peak(r1) - 0.6) < 0.02)
        with contextlib.redirect_stderr(io.StringIO()):   # the handler prints the traceback of the failure it reports
            e = handler({"input": {**base, "reference_audio_base64": junk}})
        check("ALLOW_REQUEST_REF=1: an undecodable reference returns an error, not a crash", "error" in e, str(e))
        ALLOW_REQUEST_REF = False

        # --- 4. contract and formats
        check("warmup ping", handler({"input": {"warmup": True}}) == {"ok": True})
        check("empty text is an error", "error" in handler({"input": {"text": "  "}}))
        check("unsupported language is an error", "error" in handler({"input": {**base, "language": "xx"}}))
        check("unsupported output_format is an error", "error" in handler({"input": {**base, "output_format": "mp3"}}))
        n_words = len(base["text"].split())
        n = int(24000 * 0.1 * n_words)
        w = handler({"input": {**base, "output_format": "wav"}})
        data, wsr = sf.read(io.BytesIO(base64.b64decode(w["audio_base64"])), dtype="float32")
        check("wav (default): a WAV at the model rate", w["format"] == "wav" and w["sample_rate"] == wsr == 24000 and len(data) == n)
        p = handler({"input": {**base, "output_format": "pcm16"}})
        check("pcm16: raw 16-bit samples at the model rate", p["format"] == "pcm16" and p["sample_rate"] == 24000 and len(base64.b64decode(p["audio_base64"])) == 2 * n)
        ul = handler({"input": {**base, "output_format": "ulaw8k"}})
        check("ulaw8k: 8 kHz, one byte per sample, a third of the samples",
              ul["format"] == "ulaw8k" and ul["sample_rate"] == 8000 and len(base64.b64decode(ul["audio_base64"])) == n // 3, str((len(base64.b64decode(ul["audio_base64"])), n // 3)))
        check("response fields", set(ul) == {"audio_base64", "format", "sample_rate", "language", "voice", "timing"} and
              set(ul["timing"]) == {"generate_s", "total_s", "audio_s"})

        # --- 5. the voice object of the REAL model refuses deepcopy (graph tensors): the voice cache must not depend on it. The first v3 image deep-copied it and
        #        every worker crashed at start-up with "Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol".
        try:
            MODEL, VOICES, _active_voice = FakeModel(graph=True), {}, None
            init_voices()
            r1 = handler({"input": {**base, "voice": "utkarsh", "exaggeration": 0.9}})        # generate() rebinds .t3 on the ACTIVE copy
            r2 = handler({"input": {**base, "voice": "priya"}})
            r3 = handler({"input": {**base, "voice": "utkarsh"}})
            stored = VOICES["utkarsh"]
            check("real-like voice object that refuses deepcopy: both voices prepare at start-up and every request is answered",
                  all("error" not in r for r in (r1, r2, r3)) and abs(peak(r1) - 0.2) < 0.02 and abs(peak(r2) - 0.6) < 0.02 and abs(peak(r3) - 0.2) < 0.02)
            check("a request that changes the exaggeration does not alter the stored voice, nor the next request's voice",
                  stored.t3[0] == "t3" and MODEL.conds.t3[0] == "t3", str((stored.t3, MODEL.conds.t3)))
            check("the stored voice's reference dict is not shared with the active copy", "_touched" not in stored.gen and "_touched" in MODEL.conds.gen)
        except Exception as e:                                                                  # noqa: BLE001: a crash IS the failure this case exists to catch
            check("real-like voice object that refuses deepcopy: start-up and requests work", False, f"{type(e).__name__}: {str(e)[:110]}")

    # --- 6. fail-closed loading and the streaming refusal (the pinned-runtime checks themselves are exercised in tests/test_pinned_runtime.py)
    check("load: the V3 T3 file is the pinned one and the V2 file is not", pins.T3_FILE == "t3_mtl23ls_v3.safetensors" and pins.T3_FILE in pins.CHECKPOINT_FILES
          and pins.V2_T3_FILE not in pins.CHECKPOINT_FILES)
    check("load: the checkpoint revision is a full commit hash, not a branch name", len(pins.CHECKPOINT_REVISION) == 40 and pins.CHECKPOINT_REVISION != "main")
    check("load: the 'model loaded' line names the t3 file used", loaded_line(1.234, pins.T3_FILE) == "[TTS] model loaded in 1.23s (t3_model=t3_mtl23ls_v3.safetensors)")
    try:
        pinned_runtime.load_verified_model("cuda", "/nonexistent", {"T3_MODEL": "v2"}, lambda m: None)
        check("load: T3_MODEL=v2 is refused", False, "no error")
    except pinned_runtime.PinnedRuntimeError as e:
        check("load: T3_MODEL=v2 is refused", "refused" in str(e), str(e))
    check("streaming is disabled by a constant, not by an environment variable", pins.STREAMING_ALLOWED is False)
    with tempfile.TemporaryDirectory() as d:
        MODEL, _torch, VOICES, _active_voice = FakeModel(), None, {}, None
        sf.write(os.path.join(d, "ref.wav"), 0.2 * np.sin(2 * np.pi * 200 * np.arange(8000) / 16000), 16000)
        DEFAULT_REF_WAV = os.path.join(d, "ref.wav")
        VOICES_DIR = os.path.join(d, "none")
        log = lambda m: None   # noqa: E731
        init_voices()
        for flag in (True, 1, "true"):
            r = handler({"input": {"text": "यह एक परीक्षण है", "language": "hi", "stream": flag}})
            check(f"input.stream={flag!r} is refused and carries no audio", r.get("error") == "streaming_disabled" and "audio_base64" not in r, str(r))
        r = handler({"input": {"text": "यह एक परीक्षण है", "language": "hi", "stream": False}})
        check("input.stream=false still returns the full (watermark-path) waveform", "audio_base64" in r and "error" not in r)
    ok = all(results)
    print(f"{sum(results)}/{len(results)} checks passed" + ("" if ok else "  -> FAILED"))
    return 0 if ok else 1


def main() -> int:
    if "--self-test" in sys.argv or os.getenv("SELF_TEST") == "1":
        return run_self_test()
    import runpod
    load_model()
    init_voices()
    warm_up()
    if PROFILE:    # after verification and warm-up, so the checked model is the one that gets wrapped, and the warm-up is not recorded
        install_profiling(MODEL, _torch)
    log(f"[TTS] pace: en={PACE['en']:.2f} hi={PACE['hi']:.2f} other={PACE['other']:.2f} (pitch-preserving time-stretch after synthesis; 1.00 = untouched)")
    log(f"[TTS] ready: voices={sorted(VOICES)} default={DEFAULT_VOICE} sample_rate={MODEL.sr} formats={','.join(OUTPUT_FORMATS)} "
        f"request_reference={'allowed' if ALLOW_REQUEST_REF else 'ignored'}")
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())
