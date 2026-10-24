import base64
import hashlib
import io
import os
import tempfile
import threading
import time
import traceback

import numpy as np
import runpod
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Reference voice baked into the image (Utkarsh clone). Override with env var.
DEFAULT_REF_WAV = os.getenv("DEFAULT_REF_WAV", "/app/ref_voice.wav")
WARMUP_TEXT = "नमस्ते, कैसे हैं आप?"
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "600"))
MAX_REF_BYTES = 15 * 1024 * 1024

ALLOWED_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr",
    "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
    "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}

GEN_LOCK = threading.Lock()
_current_ref_hash = None


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Load the model ONCE at import time (not inside the request)
# --------------------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_t = time.perf_counter()
MODEL = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
log(f"[TTS] model loaded in {time.perf_counter() - _t:.2f}s")


def use_reference(ref_bytes: bytes) -> float:
    """Compute voice conditionals only if the reference changed. Returns seconds spent."""
    global _current_ref_hash
    ref_hash = hashlib.sha1(ref_bytes).hexdigest()
    if ref_hash == _current_ref_hash:
        return 0.0
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(ref_bytes)
        f.flush()
        MODEL.prepare_conditionals(f.name)
    _current_ref_hash = ref_hash
    return time.perf_counter() - t0


def generate_audio(text, language, **gen_kwargs):
    with torch.inference_mode():
        wav = MODEL.generate(text, language_id=language, **gen_kwargs)
    torch.cuda.synchronize()
    return wav.detach().float().cpu().squeeze().numpy()


# Precompute the default voice once, then warm up CUDA kernels
if os.path.exists(DEFAULT_REF_WAV):
    with open(DEFAULT_REF_WAV, "rb") as fh:
        spent = use_reference(fh.read())
    log(f"[TTS] default reference prepared in {spent:.2f}s")
    _t = time.perf_counter()
    generate_audio(WARMUP_TEXT, "hi")
    generate_audio(WARMUP_TEXT, "hi")
    log(f"[TTS] warmup done in {time.perf_counter() - _t:.2f}s")
else:
    log(f"[TTS] WARNING: default reference not found at {DEFAULT_REF_WAV}")


def encode_audio(audio: np.ndarray, sr: int, output_format: str) -> str:
    if output_format == "pcm16":
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        return base64.b64encode(pcm.tobytes()).decode("ascii")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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

    output_format = data.get("output_format", "wav")  # "wav" (default) or "pcm16"
    if output_format not in ("wav", "pcm16"):
        return {"error": "output_format must be 'wav' or 'pcm16'"}

    # Optional generation knobs (only passed when provided)
    gen_kwargs = {}
    for key in ("exaggeration", "cfg_weight", "temperature"):
        if key in data:
            gen_kwargs[key] = float(data[key])

    try:
        with GEN_LOCK:
            ref_s = 0.0
            ref_b64 = data.get("reference_audio_base64")
            if ref_b64:
                ref_bytes = base64.b64decode(ref_b64, validate=True)
                if len(ref_bytes) > MAX_REF_BYTES:
                    return {"error": "Reference audio must be 15 MB or smaller"}
                ref_s = use_reference(ref_bytes)  # no-op if same voice as before
            elif _current_ref_hash is None:
                return {"error": "No reference voice available. Bake one into the image "
                                 "or send reference_audio_base64"}

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
            f"chars={len(text)} audio_s={audio_s:.2f} ref_s={ref_s:.3f} "
            f"generate_s={generate_s:.3f} encode_s={encode_s:.3f} "
            f"total_s={total_s:.3f} rtf={generate_s / max(audio_s, 1e-6):.2f}"
        )

        return {
            "audio_base64": audio_b64,
            "format": output_format,
            "sample_rate": MODEL.sr,
            "language": language,
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


runpod.serverless.start({"handler": handler})
