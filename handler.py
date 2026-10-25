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
import io
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

GEN_LOCK = threading.Lock()
MODEL = None            # the Chatterbox model (a fake one in --self-test)
_torch = None           # set by load_model(); None means "no torch": generate_audio then uses the model's output as it is (self-test)
VOICES = {}             # voice name -> prepared conditionals, built ONCE per worker at start-up
_active_voice = None


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------------------------- model and voices
def load_model():
    """Load the model ONCE at start-up (never inside a request). Needs a CUDA GPU."""
    global MODEL, _torch
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t = time.perf_counter()
    MODEL = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    _torch = torch
    log(f"[TTS] model loaded in {time.perf_counter() - t:.2f}s")


def prepare_voice(name: str, wav_path: str) -> float:
    """Prepare one voice (the slow part of Chatterbox) and keep its conditionals under `name`. Returns the seconds it took."""
    t0 = time.perf_counter()
    MODEL.prepare_conditionals(wav_path)
    conds = getattr(MODEL, "conds", None)
    if conds is None:   # fail at start-up, loudly, instead of serving the wrong voice
        raise RuntimeError("the model has no .conds after prepare_conditionals(): this Chatterbox version cannot cache voices by name")
    VOICES[name] = copy.deepcopy(conds)
    return time.perf_counter() - t0


def select_voice(name: str) -> None:
    """Make `name` the active voice. A dictionary lookup: no audio is read and nothing is prepared. Call inside GEN_LOCK."""
    global _active_voice
    if name != _active_voice:
        MODEL.conds = copy.deepcopy(VOICES[name])
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
        return np.atleast_1d(np.asarray(MODEL.generate(text, language_id=language, **gen_kwargs), dtype=np.float32))
    with _torch.inference_mode():
        wav = MODEL.generate(text, language_id=language, **gen_kwargs)
    _torch.cuda.synchronize()
    return np.atleast_1d(wav.detach().float().cpu().squeeze().numpy())


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
        return {"ok": True}

    total_start = time.perf_counter()

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"error": "Provide a non-empty input.text"}
    text = text.strip()[:MAX_TEXT_CHARS]

    language = data.get("language", "hi")
    if language not in ALLOWED_LANGUAGES:
        return {"error": f"Unsupported language: {language}"}

    output_format = data.get("output_format", "wav")  # "wav" (default), "pcm16" (model rate) or "ulaw8k" (8 kHz mu-law)
    if output_format not in OUTPUT_FORMATS:
        return {"error": "output_format must be 'wav', 'pcm16' or 'ulaw8k'"}
    if output_format == "ulaw8k" and MODEL.sr % 8000:
        return {"error": f"ulaw8k needs a model rate that is a multiple of 8000 (got {MODEL.sr})"}

    voice = data.get("voice", DEFAULT_VOICE)
    if voice not in VOICES and not (ALLOW_REQUEST_REF and data.get("reference_audio_base64")):
        return {"error": f"Unknown voice: {voice!r}", "voices": sorted(VOICES)}

    # Optional generation knobs (only passed when provided)
    gen_kwargs = {}
    for key in ("exaggeration", "cfg_weight", "temperature"):
        if key in data:
            gen_kwargs[key] = float(data[key])

    try:
        with GEN_LOCK:
            ref_s = 0.0
            ref_b64 = data.get("reference_audio_base64") if ALLOW_REQUEST_REF else None
            if ref_b64:   # legacy path (off by default): a voice sent in the request is prepared on EVERY such request (slow)
                ref_bytes = base64.b64decode(ref_b64, validate=True)
                if len(ref_bytes) > MAX_REF_BYTES:
                    return {"error": "Reference audio must be 15 MB or smaller"}
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    f.write(ref_bytes)
                    f.flush()
                    ref_s = prepare_voice("request", f.name)
                voice = "request"
            select_voice(voice)   # by name: no reference audio is read or prepared per request

            t0 = time.perf_counter()
            audio = generate_audio(text, language, **gen_kwargs)
            generate_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        audio_b64 = encode_audio(audio, MODEL.sr, output_format)
        encode_s = time.perf_counter() - t0

        audio_s = len(audio) / MODEL.sr
        total_s = time.perf_counter() - total_start
        log(
            "[TTS_TIMING] "
            f"chars={len(text)} audio_s={audio_s:.2f} ref_s={ref_s:.3f} voice={voice} "
            f"generate_s={generate_s:.3f} encode_s={encode_s:.3f} "
            f"total_s={total_s:.3f} rtf={generate_s / max(audio_s, 1e-6):.2f} fmt={output_format}"
        )

        return {
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

    except Exception as exc:
        traceback.print_exc()
        log(f"[TTS_ERROR] {exc}")
        return {"error": str(exc), "type": type(exc).__name__}


# ---------------------------------------------------------------------------------------------------------- self-test (no GPU)
def run_self_test() -> int:
    """`python handler.py --self-test`: checks voice caching, the request contract, the three output formats and the mu-law encoder with a FAKE
    model. Needs numpy, scipy and soundfile only: no torch, no chatterbox, no GPU, no network. Exit code 0 = all passed."""
    global MODEL, _torch, VOICES, _active_voice, DEFAULT_REF_WAV, DEFAULT_VOICE, VOICES_DIR, ALLOW_REQUEST_REF, log
    results, lines = [], []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""), flush=True)

    class FakeModel:
        sr = 24000

        def __init__(self):
            self.conds, self.prepares = None, []

        def prepare_conditionals(self, path):
            data, _ = sf.read(path, dtype="float32")
            self.prepares.append(os.path.basename(path))
            self.conds = {"amp": float(np.abs(data).max())}      # the "voice" of the fake model is its loudness

        def generate(self, text, language_id="hi", **kw):
            n = int(self.sr * 0.1 * max(1, len(text.split())))
            t = np.arange(n) / self.sr
            return (self.conds["amp"] * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

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
    log(f"[TTS] ready: voices={sorted(VOICES)} default={DEFAULT_VOICE} sample_rate={MODEL.sr} formats={','.join(OUTPUT_FORMATS)} "
        f"request_reference={'allowed' if ALLOW_REQUEST_REF else 'ignored'}")
    runpod.serverless.start({"handler": handler})
    return 0


if __name__ == "__main__":
    sys.exit(main())
