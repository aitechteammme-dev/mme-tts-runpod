"""GPU-host validation of real Chatterbox Multilingual V3 incremental streaming.  STATUS: IMPLEMENTED, NOT VALIDATED (never run: no GPU where it was written).

Run on a CUDA host that has the pinned Chatterbox install, the V3 checkpoint and a reference voice:
  python -m scripts.gpu_validate_streaming --voice-wav /path/ref.wav --out /tmp/stream_val [--seeds 5] [--quick]

Stages (each prints a JSON block; the final file is <out>/report.json; WAVs for LISTENING are in <out>/wav):
  0 env        versions, GPU, and `verify_installed_source()` (the adaptors copy upstream code: a mismatch stops the run).
  1 t3_equiv   same seed, same inputs: streamed tokens must equal drop_invalid_tokens(T3.inference(...)). Any mismatch is a FAILURE to explain, not tolerate.
  2 decoder    full decode vs full-prefix streaming vs bounded windows over a (context, hop, first_hop, crossfade) grid: first-audio time, total time, peak GPU
               memory, length error, log-mel distance to the full decode COMPARED WITH THE NOISE FLOOR (the CFM draws new noise each call, so two full decodes of
               the same tokens already differ), and a boundary-click ratio. Nothing is auto-approved: pick a configuration only after LISTENING.
  3 latency    real T3 -> window decoder -> ulaw8k generator vs `generate()`: time to first token / first chunk / end, RTF, cancel latency.
Acceptance is NOT decided here. The numbers are inputs to a listening test (pronunciation, speaker similarity, prosody, pauses, boundary artifacts).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

PROMPTS = {
    "hi_short": ("hi", "जी, बिल्कुल।"),
    "hi_medium": ("hi", "जी, B.Tech में आपका स्वागत है। आप किस branch में admission लेना चाहते हैं?"),
    "en_medium": ("en", "Thank you for calling. Which course are you interested in this year?"),
    "hinglish_long": ("hi", "जी, हमारे पास Computer Science, AI और Mechanical Engineering courses हैं, और merit के आधार पर scholarship भी मिलती है। "
                            "क्या मैं आपको fee structure के बारे में बताऊँ?"),
}


def logmel(x: np.ndarray, sr: int = 24000) -> np.ndarray:
    import librosa
    return np.log(librosa.feature.melspectrogram(y=x.astype(np.float32), sr=sr, n_fft=1024, hop_length=256, n_mels=64) + 1e-5)


def mel_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    ma, mb = logmel(a[:n]), logmel(b[:n])
    return float(np.mean(np.abs(ma - mb)))


def click_ratio(audio: np.ndarray, boundaries: list[int]) -> float:
    """Largest sample-to-sample jump at any chunk boundary divided by the median jump of the whole signal (1.0 = boundaries look like the rest)."""
    d = np.abs(np.diff(audio))
    med = float(np.median(d)) or 1e-9
    return float(max((d[b - 1] for b in boundaries if 0 < b < len(d)), default=0.0) / med)


def env_report(model) -> dict:
    import importlib.metadata as md

    import torch
    try:
        from streaming.chatterbox_adapter import UPSTREAM_REVISION, verify_installed_source
    except ImportError:
        try:
            from tools.tts_stream_worker.chatterbox_adapter import UPSTREAM_REVISION, verify_installed_source
        except ImportError:
            from chatterbox_adapter import UPSTREAM_REVISION, verify_installed_source
    mism = verify_installed_source()
    rep = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
           "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1), "chatterbox_dist": _ver(md, "chatterbox-tts"),
           "expected_upstream": UPSTREAM_REVISION, "source_mismatches": mism, "t3_model_env": os.getenv("T3_MODEL", "v3"), "model_sr": model.sr,
           "meanflow": bool(getattr(model.s3gen, "meanflow", False))}
    if mism:
        raise SystemExit(f"STOP: installed Chatterbox differs from the revision the adaptors were copied from: {mism}\n{json.dumps(rep, indent=2)}")
    return rep


def _ver(md, name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def stage_t3_equiv(model, voice, seeds: int) -> dict:
    import torch
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    try:
        from streaming.chatterbox_adapter import prepare_text_tokens, t3_token_stream
    except ImportError:
        try:
            from tools.tts_stream_worker.chatterbox_adapter import prepare_text_tokens, t3_token_stream
        except ImportError:
            from chatterbox_adapter import prepare_text_tokens, t3_token_stream
    model.prepare_conditionals(voice)
    rows = []
    for name, (lang, text) in PROMPTS.items():
        tt = prepare_text_tokens(model, text, lang)
        for seed in range(seeds):
            kw = dict(temperature=0.8, cfg_weight=0.5, repetition_penalty=1.2, min_p=0.05, top_p=1.0)
            torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)
            ref = drop_invalid_tokens(model.t3.inference(t3_cond=model.conds.t3, text_tokens=tt, max_new_tokens=1000, **kw)[0]).tolist()
            torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)
            got = list(t3_token_stream(model.t3, t3_cond=model.conds.t3, text_tokens=tt, max_new_tokens=1000, **kw))
            first_bad = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), None if len(ref) == len(got) else min(len(ref), len(got)))
            rows.append({"prompt": name, "seed": seed, "n_ref": len(ref), "n_stream": len(got), "identical": first_bad is None, "first_mismatch": first_bad})
    return {"all_identical": all(r["identical"] for r in rows), "rows": rows}


def stage_decoder(model, voice, out_dir, quick: bool) -> dict:
    import soundfile as sf
    import torch
    try:
        from streaming.chatterbox_adapter import SAMPLES_PER_TOKEN, make_window_decoder, prepare_text_tokens, t3_token_stream
        from streaming.stream_core import StreamDecoder
    except ImportError:
        try:
            from tools.tts_stream_worker.chatterbox_adapter import SAMPLES_PER_TOKEN, make_window_decoder, prepare_text_tokens, t3_token_stream
            from tools.tts_stream_worker.stream_core import StreamDecoder
        except ImportError:
            from chatterbox_adapter import SAMPLES_PER_TOKEN, make_window_decoder, prepare_text_tokens, t3_token_stream
            from stream_core import StreamDecoder
    model.prepare_conditionals(voice)
    os.makedirs(os.path.join(out_dir, "wav"), exist_ok=True)
    if quick:
        grid = [(25, 12, 4, 240), (50, 12, 4, 240)]
    else:
        grid = list(itertools.product((10, 25, 50), (8, 12, 24), (4, 8), (0, 240, 480)))
    rows = []
    for name, (lang, text) in PROMPTS.items():
        torch.manual_seed(0), torch.cuda.manual_seed_all(0)
        tt = prepare_text_tokens(model, text, lang)
        tokens = list(t3_token_stream(model.t3, t3_cond=model.conds.t3, text_tokens=tt, max_new_tokens=1000, top_p=1.0))
        dec = make_window_decoder(model.s3gen, model.conds.gen)
        full_a, full_b = dec(tokens, True), dec(tokens, True)           # two independent full decodes = the noise floor
        floor = mel_distance(full_a, full_b)
        sf.write(os.path.join(out_dir, "wav", f"{name}_full.wav"), full_a, model.sr)

        def run(label, **kw):
            torch.cuda.reset_peak_memory_stats()
            sd = StreamDecoder(dec, SAMPLES_PER_TOKEN, lookahead_tokens=3, **kw)
            t0, first, chunks, bounds, total = time.perf_counter(), None, [], [], 0
            for t in tokens:
                for piece in sd.push([t]):
                    first = first if first is not None else time.perf_counter() - t0
                    chunks.append(piece)
                    total += len(piece)
                    bounds.append(total)
            for piece in sd.flush():
                chunks.append(piece)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
            sf.write(os.path.join(out_dir, "wav", f"{name}_{label}.wav"), audio, model.sr)
            return {"prompt": name, "mode": label, "tokens": len(tokens), "first_audio_s_after_push_start": first, "decode_wall_s": round(wall, 3),
                    "decoded_tokens": sd.stats["decoded_tokens"], "max_window": sd.stats["max_window"],
                    "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), "len_error_samples": len(audio) - len(full_a),
                    "mel_dist_to_full": round(mel_distance(audio, full_a), 4), "noise_floor": round(floor, 4),
                    "click_ratio": round(click_ratio(audio, bounds[:-1]), 2)}
        rows.append(run("fullprefix", context_tokens=None, hop_tokens=12, first_hop_tokens=4, crossfade_samples=240))
        for ctx, hop, fh, xf in grid:
            if ctx * SAMPLES_PER_TOKEN >= xf:
                rows.append(run(f"win_c{ctx}_h{hop}_f{fh}_x{xf}", context_tokens=ctx, hop_tokens=hop, first_hop_tokens=fh, crossfade_samples=xf))
    return {"note": "mel_dist_to_full must be judged against noise_floor; then LISTEN. Reject configurations with a clearly worse sound even if faster.", "rows": rows}


def stage_latency(model, voice, runs: int) -> dict:
    import torch
    try:
        from streaming.chatterbox_adapter import (hold_back_last_token, make_window_decoder, prepare_text_tokens, t3_token_stream)
        from streaming.worker_stream import stream_synth
    except ImportError:
        try:
            from tools.tts_stream_worker.chatterbox_adapter import (hold_back_last_token, make_window_decoder, prepare_text_tokens, t3_token_stream)
            from tools.tts_stream_worker.worker_stream import stream_synth
        except ImportError:
            from chatterbox_adapter import (hold_back_last_token, make_window_decoder, prepare_text_tokens, t3_token_stream)
            from worker_stream import stream_synth
    model.prepare_conditionals(voice)
    out = []
    for name, (lang, text) in PROMPTS.items():
        for _ in range(runs):
            t0 = time.perf_counter()
            wav = model.generate(text, language_id=lang)
            torch.cuda.synchronize()
            t_full, audio_s = time.perf_counter() - t0, wav.shape[-1] / model.sr
            tt = prepare_text_tokens(model, text, lang)
            marks, n, t0 = {}, 0, time.perf_counter()

            def tokens():
                for tok in hold_back_last_token(t3_token_stream(model.t3, t3_cond=model.conds.t3, text_tokens=tt, max_new_tokens=1000)):
                    marks.setdefault("first_token_s", time.perf_counter() - t0)
                    yield tok
            for item in stream_synth(tokens(), lambda: make_window_decoder(model.s3gen, model.conds.gen)):
                n += 1
                marks.setdefault("first_chunk_s", time.perf_counter() - t0)
            marks["stream_total_s"] = time.perf_counter() - t0
            out.append({"prompt": name, "generate_total_s": round(t_full, 3), "audio_s": round(audio_s, 2), "chunks": n,
                        **{k: round(v, 3) for k, v in marks.items()}, "stream_rtf": round(marks["stream_total_s"] / audio_s, 2)})
    # cancel latency: stop after 15 tokens, measure how long until the generator has finished and released the GPU loop
    lang, text = PROMPTS["hinglish_long"]
    tt = prepare_text_tokens(model, text, lang)
    state = {"n": 0, "t_cancel": None}

    def should_cancel():
        state["n"] += 1
        if state["n"] == 15 and state["t_cancel"] is None:
            state["t_cancel"] = time.perf_counter()
        return state["n"] >= 15
    list(stream_synth(t3_token_stream(model.t3, t3_cond=model.conds.t3, text_tokens=tt, should_cancel=should_cancel),
                      lambda: make_window_decoder(model.s3gen, model.conds.gen), should_cancel=should_cancel))
    cancel_s = time.perf_counter() - state["t_cancel"]
    return {"rows": out, "cancel_latency_s": round(cancel_s, 4), "note": "cancel latency here is in-process; the RunPod cancel path must be measured separately"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice-wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    if not torch.cuda.is_available():
        print("refusing to run: no CUDA GPU", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    report = {"env": env_report(model)}
    for name, fn in (("t3_equiv", lambda: stage_t3_equiv(model, a.voice_wav, a.seeds)),
                     ("decoder", lambda: stage_decoder(model, a.voice_wav, a.out, a.quick)),
                     ("latency", lambda: stage_latency(model, a.voice_wav, a.runs))):
        report[name] = fn()
        print(json.dumps({name: report[name]}, indent=1, ensure_ascii=False)[:4000])
        with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        if name == "t3_equiv" and not report[name]["all_identical"]:
            print("STOP: streamed tokens differ from T3.inference. Explain the cause before going on (RNG use, cache init, stop handling).", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
