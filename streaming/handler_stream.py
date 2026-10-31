"""Unified RunPod streaming generator handler for Chatterbox Multilingual V3 incremental streaming.

Connects:
  - `chatterbox_adapter.py` (t3_token_stream, prepare_text_tokens, make_window_decoder)
  - `stream_core.py` (StreamDecoder with bounded context and crossfading)
  - `worker_stream.py` (stream_synth, StreamingDecimator, pcm16_to_ulaw)

Yields contract per item:
  {"seq": int, "audio_base64": str, "format": "ulaw8k", "sample_rate": 8000, "final": bool}

Watermarking Note:
  Chatterbox V3 applies Resemble Perth watermarking to the complete finished waveform only.
  Incremental streaming emits audio before T3/S3Gen finishes; chunked Perth watermarking
  does not exist upstream. A policy gate `ALLOW_UNWATERMARKED_STREAMING` is provided.
"""
from __future__ import annotations

import base64
import os
import sys
import time
from typing import Any, Callable, Iterator

try:
    from streaming.chatterbox_adapter import (
        hold_back_last_token,
        make_window_decoder,
        prepare_text_tokens,
        require_matching_revision,
        t3_token_stream,
    )
    from streaming.worker_stream import stream_synth
except ImportError:
    try:
        from tools.tts_stream_worker.chatterbox_adapter import (
            hold_back_last_token,
            make_window_decoder,
            prepare_text_tokens,
            require_matching_revision,
            t3_token_stream,
        )
        from tools.tts_stream_worker.worker_stream import stream_synth
    except ImportError:
        from chatterbox_adapter import (  # type: ignore
            hold_back_last_token,
            make_window_decoder,
            prepare_text_tokens,
            require_matching_revision,
            t3_token_stream,
        )
        from worker_stream import stream_synth  # type: ignore

MAX_TEXT_CHARS = 400
ALLOWED_LANGUAGES = {"hi", "en", "mr", "ta", "te", "kn", "gu", "bn"}
ALLOW_UNWATERMARKED_STREAMING = os.getenv("ALLOW_UNWATERMARKED_STREAMING", "0") == "1"


def handler_stream_generator(job: dict, model: Any, voices: dict, default_voice: str = "utkarsh",
                             gen_lock: Any = None) -> Iterator[dict]:
    """Generator handler that yields progressive audio chunks for RunPod Serverless /stream/{id}.
    
    Session isolation:
      Each invocation creates isolated StreamDecoder and StreamingDecimator instances.
      Voice conditioning is read-only (shallow copied).
    """
    data = job.get("input", {}) or {}

    # 1. Warmup ping
    if data.get("warmup"):
        yield {"seq": 0, "audio_base64": "", "format": "ulaw8k", "sample_rate": 8000, "final": True, "ok": True}
        return

    # 2. Watermark policy gate
    if not ALLOW_UNWATERMARKED_STREAMING and not data.get("allow_unwatermarked"):
        yield {
            "seq": 0,
            "error": "streaming_disabled",
            "detail": "Streamed audio cannot carry the Perth watermark (requires complete finished waveform). Set ALLOW_UNWATERMARKED_STREAMING=1 or omit stream:true for complete watermarked WAV.",
            "final": True
        }
        return

    # 3. Input validation
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        yield {"seq": 0, "error": "Provide a non-empty input.text", "final": True}
        return
    text = text.strip()[:MAX_TEXT_CHARS]

    language = data.get("language", "hi")
    if language not in ALLOWED_LANGUAGES:
        yield {"seq": 0, "error": f"Unsupported language: {language}", "final": True}
        return

    voice_name = data.get("voice", default_voice)
    if voice_name not in voices:
        yield {"seq": 0, "error": f"Unknown voice: {voice_name!r}", "voices": sorted(voices.keys()), "final": True}
        return

    # 4. Voice selection & conditionals
    voice_conds = voices[voice_name]
    t3_cond = voice_conds.get("t3")
    gen_conds = voice_conds.get("gen", {})

    gen_kwargs = {
        "temperature": float(data.get("temperature", 0.8)),
        "cfg_weight": float(data.get("cfg_weight", 0.5)),
        "repetition_penalty": float(data.get("repetition_penalty", 1.2)),
        "min_p": float(data.get("min_p", 0.05)),
        "top_p": float(data.get("top_p", 1.0)),
        "max_new_tokens": int(data.get("max_new_tokens", 1000)),
    }

    # 5. Token preparation
    text_tokens = prepare_text_tokens(model, text, language)

    # 6. Setup T3 speech token stream wrapped in hold_back_last_token
    token_src = hold_back_last_token(
        t3_token_stream(
            model.t3,
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            **gen_kwargs
        )
    )

    # 7. S3Gen window decoder factory
    decode_factory = lambda: make_window_decoder(model.s3gen, gen_conds)

    # 8. Progressive streaming synthesis
    yield from stream_synth(
        token_source=token_src,
        decode_factory=decode_factory,
        model_rate=model.sr,
        samples_per_token=960,
        lookahead_tokens=3,
        hop_tokens=12,
        first_hop_tokens=4,
        context_tokens=25,
        crossfade_samples=240,
    )

