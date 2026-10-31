"""Worker-side generator for progressive TTS: speech tokens in, ordered audio chunks out (numpy only; no torch, no chatterbox, no runpod).

STATUS: logic verified with fakes (tests/unit/test_tts_worker_stream.py). The GPU parts are INJECTED and NOT implemented here:
  * `token_source`  - an iterator of S3 speech tokens from a streaming T3 loop (T3.inference turned into a generator; needs the pinned
                      Chatterbox checkout and a GPU),
  * `decode`        - the S3Gen window decoder passed to `StreamDecoder`.
RunPod wiring (also NOT done, needs the worker repo + a GPU host): `runpod.serverless.start({"handler": handler_gen})` where `handler_gen`
is a generator that `yield`s the dicts below; `/stream/{id}` then returns them. Item contract (parsed by `ModelClients.synthesize_stream`):
  {"seq": int from 0, "audio_base64": str, "format": "ulaw8k", "sample_rate": 8000, "final": bool}   (the last item is `final` and may be empty)
Watermarking: Perth works on a complete waveform, so streamed audio is NOT watermarked (same as the Turbo reference). Product decision needed.
"""
from __future__ import annotations

import base64
from typing import Callable, Iterator, Optional

import numpy as np

try:
    from streaming.stream_core import StreamDecoder
except ImportError:
    try:
        from tools.tts_stream_worker.stream_core import StreamDecoder
    except ImportError:
        from stream_core import StreamDecoder


class StreamingDecimator:
    """Stateful FIR low-pass + decimate-by-`factor` (24 kHz -> 8 kHz is 3). Feeding a signal in ANY chunking gives exactly the samples of
    feeding it whole, so chunk boundaries do not restart the filter (no clicks, no phase jumps)."""

    def __init__(self, factor: int = 3, taps: int = 63):
        n = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(n / factor) * np.hanning(taps)           # cut-off at the new Nyquist
        self._h = (h / h.sum()).astype(np.float64)
        self._factor = factor
        self._hist = np.zeros(taps - 1, dtype=np.float64)    # last taps-1 input samples
        self._phase = 0                                      # input samples to skip before the next output

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        buf = np.concatenate([self._hist, x.astype(np.float64)])
        y = np.convolve(buf, self._h, mode="valid")          # len(x) filtered samples, aligned to the inputs of x
        out = y[self._phase::self._factor]
        self._phase = (self._phase - len(x)) % self._factor   # input samples still to skip before the next output, carried to the next chunk
        self._hist = buf[-len(self._hist):]
        return out.astype(np.float32)


def pcm16_to_ulaw(pcm: np.ndarray) -> bytes:
    """G.711 mu-law (same algorithm as the worker's handler.py, which is bit-identical to audioop.lin2ulaw)."""
    seg_end = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
    v = pcm.astype(np.int32) >> 2
    mask = np.where(v < 0, 0x7F, 0xFF)
    v = np.minimum(np.abs(v), 8159) + 33
    seg = (v[:, None] > seg_end[None, :]).sum(axis=1)
    uval = np.where(seg >= 8, 0x7F, (seg << 4) | ((v >> (seg + 1)) & 0x0F))
    return (uval ^ mask).astype(np.uint8).tobytes()


def stream_synth(token_source: Iterator[int], decode_factory: Callable[[], Callable], *, model_rate: int = 24000, samples_per_token: int = 960,
                 lookahead_tokens: int = 3, hop_tokens: int = 12, first_hop_tokens: Optional[int] = 4, context_tokens: Optional[int] = 25,
                 crossfade_samples: int = 240, should_cancel: Callable[[], bool] = lambda: False) -> Iterator[dict]:
    """Generator handler body. Yields contract items as soon as audio is STABLE, never after the whole utterance is generated.
    `decode_factory()` returns a fresh S3Gen window decoder (state is per request, never shared). Closing this generator (RunPod cancel / client
    gone) closes the token source and drops the decoder state."""
    if model_rate % 8000:
        raise ValueError("ulaw8k needs a model rate that is a multiple of 8000")
    decoder = StreamDecoder(decode_factory(), samples_per_token, lookahead_tokens=lookahead_tokens, hop_tokens=hop_tokens,
                            first_hop_tokens=first_hop_tokens, context_tokens=context_tokens, crossfade_samples=crossfade_samples)
    resample = StreamingDecimator(model_rate // 8000)
    seq = 0

    def item(audio: np.ndarray, final: bool) -> dict:
        nonlocal seq
        pcm = (np.clip(resample.process(audio), -1.0, 1.0) * 32767.0).astype(np.int16)
        out = {"seq": seq, "audio_base64": base64.b64encode(pcm16_to_ulaw(pcm)).decode("ascii") if len(pcm) else "",
               "format": "ulaw8k", "sample_rate": 8000, "final": final}
        seq += 1
        return out

    try:
        for tok in token_source:
            if should_cancel():
                return
            for piece in decoder.push([tok]):
                yield item(piece, False)
        tail = decoder.flush()
        for piece in tail[:-1]:
            yield item(piece, False)
        yield item(tail[-1] if tail else np.zeros(0, np.float32), True)   # the stream always ends with ONE final item
    finally:
        decoder.close()                                  # idempotent: drops buffered tokens and held audio after flush, cancel or error
        close = getattr(token_source, "close", None)
        if close:
            close()
