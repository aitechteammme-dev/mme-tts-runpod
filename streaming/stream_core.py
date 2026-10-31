"""Framework-free core of incremental speech decoding for the Chatterbox TTS worker (numpy only: no torch, no chatterbox, no runpod).

STATUS: logic verified against a SYNTHETIC decoder (tests/unit/test_tts_stream_core.py). NOT validated against the real S3Gen on a GPU:
whether a windowed re-decode keeps Chatterbox's quality is the open research question (see docs/STREAMING_TTS_STATUS.md).

Why this exists. S3Gen turns speech tokens (25 Hz) into audio: a Conformer encoder with a small right look-ahead feeds a flow-matching decoder
and a HiFi-GAN vocoder. It is not causal, so audio for the last `lookahead` tokens is unstable until more tokens arrive. `StreamDecoder`
therefore only emits audio for tokens whose right context is known, re-decodes a BOUNDED window, and joins the pieces with a cross-fade.

The model is injected as `decode(window_tokens, final) -> float32 samples`:
  * `final=False`: returns audio for len(window) - lookahead tokens (the real `flow_inference(finalize=False)` ignores the last 3 tokens);
  * `final=True`:  returns audio for every token in the window.
Two modes:
  * `context_tokens=N`  (bounded): each decode sees at most N already-emitted tokens + the new ones. Cost per push is constant.
  * `context_tokens=None` (full prefix): the whole history is decoded every time, as the Turbo reference streamer does. Quality-safe but
    QUADRATIC; it exists to A/B quality against the bounded mode, not as the production setting.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

DecodeFn = Callable[[Sequence[int], bool], np.ndarray]


class StreamClosed(RuntimeError):
    """push()/flush() after close() (cancellation) or after flush()."""


class StreamDecoder:
    def __init__(self, decode: DecodeFn, samples_per_token: int, *, lookahead_tokens: int = 3, hop_tokens: int = 12,
                 first_hop_tokens: Optional[int] = None, context_tokens: Optional[int] = 25, crossfade_samples: int = 0):
        if samples_per_token <= 0 or lookahead_tokens < 0 or hop_tokens <= 0 or crossfade_samples < 0:
            raise ValueError("invalid streaming parameters")
        if context_tokens is not None and context_tokens * samples_per_token < crossfade_samples:
            raise ValueError("context must cover the cross-fade: context_tokens * samples_per_token >= crossfade_samples")
        self._decode, self._spt, self._look = decode, samples_per_token, lookahead_tokens
        self._hop = hop_tokens
        self._first_hop = first_hop_tokens if first_hop_tokens is not None else hop_tokens   # a smaller first hop = earlier first audio
        self._ctx, self._xf = context_tokens, crossfade_samples
        self._tokens: List[int] = []     # buffered tokens; absolute index of _tokens[0] is _base
        self._base = 0
        self._emitted_tokens = 0         # absolute count of tokens whose audio has been handed out or is held in _tail
        self._tail = np.zeros(0, dtype=np.float32)   # last `crossfade_samples` of audio emitted-in-principle but not yet released
        self._state = "open"             # open -> flushed | closed
        self.stats = {"decodes": 0, "decoded_tokens": 0, "max_window": 0}   # evidence for the bounded-cost claim

    # ------------------------------------------------------------------ public
    def push(self, tokens: Iterable[int]) -> List[np.ndarray]:
        """Add speech tokens; returns the audio chunks (possibly none) that are now stable. Order is the playback order."""
        self._require_open()
        self._tokens.extend(int(t) for t in tokens)
        out: List[np.ndarray] = []
        while True:
            total = self._base + len(self._tokens)
            ready = total - self._look - self._emitted_tokens        # tokens whose audio is stable
            need = self._first_hop if self._emitted_tokens == 0 else self._hop
            if ready < need:
                return out
            piece = self._decode_step(final=False, new_tokens=ready)
            if piece is not None and len(piece):
                out.append(piece)

    def flush(self) -> List[np.ndarray]:
        """End of utterance: decode what is left with final=True and release the held cross-fade tail. Safe for utterances shorter than a hop."""
        self._require_open()
        self._state = "flushed"
        out: List[np.ndarray] = []
        remaining = self._base + len(self._tokens) - self._emitted_tokens
        if remaining > 0:
            piece = self._decode_step(final=True, new_tokens=remaining)
            if piece is not None and len(piece):
                out.append(piece)
        if len(self._tail):
            out.append(self._tail)
            self._tail = np.zeros(0, dtype=np.float32)
        self._release()
        return out

    def close(self) -> None:
        """Cancellation: drop every buffered token and the held audio. Nothing more can be pushed; nothing is emitted."""
        self._state = "closed"
        self._release()

    # ------------------------------------------------------------------ internals
    def _require_open(self) -> None:
        if self._state != "open":
            raise StreamClosed(f"stream is {self._state}")

    def _release(self) -> None:
        self._tokens = []
        self._tail = np.zeros(0, dtype=np.float32)

    def _decode_step(self, *, final: bool, new_tokens: int) -> Optional[np.ndarray]:
        start = self._base if self._ctx is None else max(self._base, self._emitted_tokens - self._ctx)   # full prefix | bounded window
        window = self._tokens[start - self._base:]
        audio = np.asarray(self._decode(window, final), dtype=np.float32)
        self.stats["decodes"] += 1
        self.stats["decoded_tokens"] += len(window)
        self.stats["max_window"] = max(self.stats["max_window"], len(window))
        spt = self._spt
        ctx_samples = (self._emitted_tokens - start) * spt
        end_samples = ctx_samples + new_tokens * spt
        if len(audio) < end_samples:
            raise ValueError(f"decoder returned {len(audio)} samples, expected at least {end_samples}")
        # Overlap region: the cross-fade is done over the `len(tail)` samples that precede the new audio, taken from THIS decode.
        xf = len(self._tail)
        piece = audio[ctx_samples - xf:end_samples].copy() if ctx_samples >= xf else audio[ctx_samples:end_samples].copy()
        if xf and ctx_samples >= xf:
            fade = np.linspace(0.0, 1.0, xf, dtype=np.float32)
            piece[:xf] = self._tail * (1.0 - fade) + piece[:xf] * fade
        elif xf:                                                        # first decode after a full-prefix start: nothing to blend with
            piece = np.concatenate([self._tail, piece])
        self._emitted_tokens += new_tokens
        if self._xf and not final and len(piece) > self._xf:
            self._tail, piece = piece[-self._xf:].copy(), piece[:-self._xf]
        else:
            self._tail = np.zeros(0, dtype=np.float32)
        # keep only what the next window can still need
        if self._ctx is not None:
            keep_from = max(self._base, self._emitted_tokens - self._ctx)
            self._tokens = self._tokens[keep_from - self._base:]
            self._base = keep_from
        return piece
