
import os
import base64
import tempfile
import traceback
import threading
import time

import runpod
import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

MODEL = None
MODEL_LOCK = threading.Lock()


def load_model():
    global MODEL

    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA GPU is not available")

                print("[TTS] Loading Chatterbox multilingual V3...", flush=True)
                start = time.perf_counter()

                MODEL = ChatterboxMultilingualTTS.from_pretrained(
                    device="cuda",
                )

                print(
                    f"[TTS] Model loaded in "
                    f"{time.perf_counter() - start:.3f}s",
                    flush=True,
                )

    return MODEL


def handler(job):
    data = job.get("input", {})
    total_start = time.perf_counter()

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return {"error": "Provide a non-empty input.text"}

    language = data.get("language", "hi")
    allowed_languages = {
        "ar", "da", "de", "el", "en", "es", "fi", "fr",
        "he", "hi", "it", "ja", "ko", "ms", "nl", "no",
        "pl", "pt", "ru", "sv", "sw", "tr", "zh",
    }

    if language not in allowed_languages:
        return {"error": f"Unsupported language: {language}"}

    reference_b64 = data.get("reference_audio_base64")
    if not reference_b64:
        return {"error": "Provide reference_audio_base64 containing a WAV recording"}

    try:
        t0 = time.perf_counter()
        reference_bytes = base64.b64decode(reference_b64, validate=True)

        if len(reference_bytes) > 15 * 1024 * 1024:
            return {"error": "Reference audio must be 15 MB or smaller"}

        decode_s = time.perf_counter() - t0

        with tempfile.NamedTemporaryFile(suffix=".wav") as ref_file:
            t0 = time.perf_counter()
            ref_file.write(reference_bytes)
            ref_file.flush()
            reference_write_s = time.perf_counter() - t0

            model = load_model()

            with MODEL_LOCK:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.perf_counter()

                with torch.inference_mode():
                    audio = model.generate(
                        text.strip(),
                        language_id=language,
                        audio_prompt_path=ref_file.name,
                    )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                generate_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            audio = audio.detach().float().cpu().squeeze().numpy()
            cpu_transfer_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix=".wav") as out_file:
                sf.write(out_file.name, audio, model.sr)
                out_file.seek(0)

                with open(out_file.name, "rb") as generated:
                    audio_bytes = generated.read()

            wav_encode_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            base64_encode_s = time.perf_counter() - t0

            audio_duration_s = len(audio) / model.sr
            total_s = time.perf_counter() - total_start

            print(
                "[TTS_TIMING] "
                f"input_chars={len(text.strip())} "
                f"audio_s={audio_duration_s:.2f} "
                f"decode_s={decode_s:.3f} "
                f"reference_write_s={reference_write_s:.3f} "
                f"generate_s={generate_s:.3f} "
                f"cpu_transfer_s={cpu_transfer_s:.3f} "
                f"wav_encode_s={wav_encode_s:.3f} "
                f"base64_encode_s={base64_encode_s:.3f} "
                f"total_s={total_s:.3f}",
                flush=True,
            )

            return {
                "audio_base64": audio_b64,
                "format": "wav",
                "sample_rate": model.sr,
                "language": language,
            }

    except Exception as exc:
        traceback.print_exc()
        print(f"[TTS_ERROR] {exc}", flush=True)
        return {
            "error": str(exc),
            "type": type(exc).__name__,
        }


runpod.serverless.start({"handler": handler})
