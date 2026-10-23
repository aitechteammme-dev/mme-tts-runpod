
import os
import base64
import tempfile
import traceback

import runpod
import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

MODEL = None
MODEL_LOCK = __import__("threading").Lock()


def load_model():
    global MODEL

    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA GPU is not available")

                print("Loading Chatterbox multilingual V3...")
                MODEL = ChatterboxMultilingualTTS.from_pretrained(
                    device="cuda",
                )
                print("Chatterbox model loaded.")

    return MODEL


def handler(job):
    data = job.get("input", {})

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

    # Supply reference audio as base64 WAV data.
    reference_b64 = data.get("reference_audio_base64")
    if not reference_b64:
        return {
            "error": (
                "Provide reference_audio_base64 containing "
                "a base64-encoded WAV reference recording."
            )
        }

    try:
        reference_bytes = base64.b64decode(
            reference_b64,
            validate=True,
        )

        if len(reference_bytes) > 15 * 1024 * 1024:
            return {"error": "Reference audio must be 15 MB or smaller"}

        with tempfile.NamedTemporaryFile(suffix=".wav") as ref_file:
            ref_file.write(reference_bytes)
            ref_file.flush()

            model = load_model()

            with MODEL_LOCK:
                with torch.inference_mode():
                    audio = model.generate(
                        text.strip(),
                        language_id=language,
                        audio_prompt_path=ref_file.name,
                    )

            audio = audio.detach().float().cpu().squeeze().numpy()

            with tempfile.NamedTemporaryFile(suffix=".wav") as out_file:
                sf.write(out_file.name, audio, model.sr)
                out_file.seek(0)
                with open(out_file.name, "rb") as generated:
                    audio_b64 = base64.b64encode(
                        generated.read()
                    ).decode("utf-8")

            return {
                "audio_base64": audio_b64,
                "format": "wav",
                "sample_rate": model.sr,
                "language": language,
            }

    except Exception as exc:
        traceback.print_exc()
        return {
            "error": str(exc),
            "type": type(exc).__name__,
        }


runpod.serverless.start({"handler": handler})
