"""Adaptors between the real Chatterbox Multilingual model and `stream_core` / `worker_stream`.

STATUS: IMPLEMENTED, NOT VALIDATED. Written from the VERBATIM source of resemble-ai/chatterbox master @ 5de7a54 (fetched 2026-10-28). It has never been run:
this machine has no GPU/torch/chatterbox. The deployed worker's Chatterbox revision is UNKNOWN, so `verify_installed_source()` fingerprints every function
copied here and the worker must refuse to stream when the installed code differs (a different revision means this copy may be wrong).

Three things are copied/adapted, each marked where it departs from upstream:
  1. `t3_token_stream`   = the loop of `T3.inference` as a generator (same processors, same order, same RNG calls). Upstream differences: yields each
                           token instead of collecting, no tqdm/logging, honours `should_cancel`, filters tokens >= SPEECH_VOCAB_SIZE like `drop_invalid_tokens`.
  2. `flow_inference_nonfinal` = `CausalMaskedDiffWithXvec.inference` with ONE fix: for finalize=False upstream trims the encoder output `h` by
                           pre_lookahead_len*token_mel_ratio frames but still derives `h_lengths` from the UNtrimmed mask, so mask and `mu` disagree (the bug PR #528
                           fixes). Here `h_lengths` is reduced by the same amount.
  3. `make_window_decoder` = flow -> HiFT -> trim_fade, as `S3Token2Wav.inference`, for a WINDOW of tokens (reference prompt tokens are still prepended by the flow).
Known upstream facts that shape the design (all read in the source): the CFM draws fresh noise per call (`torch.randn_like`), so two decodes of the same tokens are
never sample-identical; every decode fades in the first 40 ms (`trim_fade`); `generate()` drops the audio of the LAST speech token and then watermarks the whole wav.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from typing import Callable, Iterator, List, Optional

import numpy as np

UPSTREAM_REVISION = "resemble-ai/chatterbox master 5de7a54aa4e5e2baadb0182dde554908b48b85c2"
SPEECH_VOCAB_SIZE = 6561          # tokens >= this are special (EOS ...): upstream `drop_invalid_tokens` is `x[x < SPEECH_VOCAB_SIZE]`
S3GEN_SR, S3_TOKEN_RATE = 24000, 25
SAMPLES_PER_TOKEN = S3GEN_SR // S3_TOKEN_RATE   # 960
PRE_LOOKAHEAD_TOKENS = 3

# sha256 of ast.dump() of the copied functions in UPSTREAM_REVISION (formatting/comments do not matter, code does).
EXPECTED_AST_SHA256 = {
    "T3.inference": "492f2c04caf846c02ab6ac54c71e4ce8b5f9a246d22e9d70bbc627ca2a7f925e",
    "CausalMaskedDiffWithXvec.inference": "cff99a19dea4778cced22b5561779ebacbf8cd5c18684e1776e5fdbff49fb3fe",
    "S3Token2Mel.forward": "a5d84477c0951fee564cf5fa679a368f2ba4866a9d728d393754d58c406ef41d",
    "S3Token2Wav.flow_inference": "8e5a064c172a15b65ab1234c977595c58dd997e7e382deb04cf9b5ee6cd91f1e",
    "S3Token2Wav.hift_inference": "5f622dc3c3954af21a8de31f7674853c97e56fbde5be9f7d33bcf0dfe371d978",
    "S3Token2Wav.inference": "234c9749477554b86585a4c4959635f7a9386beff500210d2b225ef5c66861ca",
}


class RevisionMismatch(RuntimeError):
    pass


def ast_fingerprint(fn) -> str:
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def verify_installed_source() -> List[str]:
    """Names of copied functions whose installed code differs from UPSTREAM_REVISION (empty list = identical). Call at worker start-up."""
    from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec
    from chatterbox.models.s3gen.s3gen import S3Token2Mel, S3Token2Wav
    from chatterbox.models.t3 import T3
    found = {"T3.inference": T3.inference, "CausalMaskedDiffWithXvec.inference": CausalMaskedDiffWithXvec.inference,
             "S3Token2Mel.forward": S3Token2Mel.forward, "S3Token2Wav.flow_inference": S3Token2Wav.flow_inference,
             "S3Token2Wav.hift_inference": S3Token2Wav.hift_inference, "S3Token2Wav.inference": S3Token2Wav.inference}
    return sorted(name for name, fn in found.items() if ast_fingerprint(fn) != EXPECTED_AST_SHA256[name])


def require_matching_revision() -> None:
    bad = verify_installed_source()
    if bad:
        raise RevisionMismatch(f"installed Chatterbox differs from {UPSTREAM_REVISION} in: {', '.join(bad)}; streaming adaptors refuse to run")


# ------------------------------------------------------------------------------------------------------------------ 1. T3 token stream
def prepare_text_tokens(model, text: str, language_id: str):
    """The text-token preparation of `ChatterboxMultilingualTTS.generate` (language check, punc_norm, CFG duplicate, SOT/EOT)."""
    import torch.nn.functional as F
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, punc_norm
    if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language_id '{language_id}'")
    text_tokens = model.tokenizer.text_to_tokens(punc_norm(text), language_id=language_id.lower() if language_id else None).to(model.device)
    text_tokens = torch_cat2(text_tokens)
    text_tokens = F.pad(text_tokens, (1, 0), value=model.t3.hp.start_text_token)
    return F.pad(text_tokens, (0, 1), value=model.t3.hp.stop_text_token)


def torch_cat2(t):
    import torch
    return torch.cat([t, t], dim=0)   # two sequences for classifier-free guidance, as upstream


def t3_token_stream(t3, *, t3_cond, text_tokens, max_new_tokens: int = 1000, temperature: float = 0.8, cfg_weight: float = 0.5,
                    repetition_penalty: float = 1.2, min_p: float = 0.05, top_p: float = 1.0,
                    should_cancel: Callable[[], bool] = lambda: False) -> Iterator[int]:
    """`T3.inference` as a generator: yields each valid speech token as soon as it is sampled. Defaults are `generate()`'s (top_p=1.0, 1000 tokens).
    Stops at EOS (not yielded) or max_new_tokens or cancel. Same KV-cache loop and RNG consumption as upstream: with the same seed the sequence
    must equal `drop_invalid_tokens(t3.inference(...)[0])` (scripts/gpu_validate_streaming.py checks this)."""
    import torch
    from chatterbox.models.t3.t3 import _ensure_BOT_EOT
    from transformers.generation.logits_process import MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper
    with torch.inference_mode():
        _ensure_BOT_EOT(text_tokens, t3.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=t3.device)
        initial_speech_tokens = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _len_cond = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=initial_speech_tokens, cfg_weight=cfg_weight)
        if not getattr(t3, "patched_model", None):   # upstream rebuilds it on every call; it only wraps shared modules, so build once per T3
            from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
            t3.patched_model = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb, speech_head=t3.speech_head)
        device = embeds.device
        bos_token = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = t3.speech_emb(bos_token) + t3.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = torch.cat([bos_embed, bos_embed])                       # batch of 2 for CFG
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        generated_ids = bos_token.clone()
        top_p_warper, min_p_warper = TopPLogitsWarper(top_p=top_p), MinPLogitsWarper(min_p=min_p)
        rep_proc = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
        output = t3.patched_model(inputs_embeds=inputs_embeds, past_key_values=None, use_cache=True, output_attentions=False,
                                  output_hidden_states=True, return_dict=True)
        past = output.past_key_values                                        # KV cache: only the NEW token is fed from now on
        for i in range(max_new_tokens):
            if should_cancel():
                return
            logits_step = output.logits[:, -1, :]
            cond, uncond = logits_step[0:1, :], logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)
            ids_for_proc = generated_ids[:1, ...]
            logits = rep_proc(ids_for_proc, logits)
            if temperature != 1.0:
                logits = logits / temperature
            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            tok = int(next_token.view(-1)[0].item())                         # one device->host sync per token (upstream already syncs on the EOS check)
            if tok == t3.hp.stop_speech_token:
                return
            if tok < SPEECH_VOCAB_SIZE:
                yield tok
            next_token_embed = t3.speech_emb(next_token) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            next_token_embed = torch.cat([next_token_embed, next_token_embed])
            output = t3.patched_model(inputs_embeds=next_token_embed, past_key_values=past, output_attentions=False,
                                      output_hidden_states=True, return_dict=True)
            past = output.past_key_values


def hold_back_last_token(tokens: Iterator[int]) -> Iterator[int]:
    """`generate()` discards the audio of the FINAL speech token (it decodes to ~40 ms of noise). A stream cannot know a token is final until EOS, so each
    token is released one step late and the last is dropped (kept only when it is the sole token, upstream's `max(1, n - 1)`). Costs 40 ms of latency.
    Difference from upstream: the dropped token is also absent from the decoder's right context."""
    held: Optional[int] = None
    n = 0
    for t in tokens:
        if held is not None:
            yield held
            n += 1
        held = t
    if held is not None and n == 0:
        yield held


# ------------------------------------------------------------------------------------------------------------------ 2. S3Gen window decoder
def flow_inference_nonfinal(flow, *, token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding,
                            finalize: bool, n_timesteps: int = 10, noised_mels=None, meanflow: bool = False):
    """`CausalMaskedDiffWithXvec.inference` (upstream lines 146-198) with the finalize=False mask fix. For finalize=True this is upstream's behaviour."""
    import torch
    import torch.nn.functional as F
    from chatterbox.models.s3gen.flow import _repeat_batch_dim
    from chatterbox.models.s3gen.utils.mask import make_pad_mask
    with torch.inference_mode():
        B = token.size(0)
        embedding = F.normalize(torch.atleast_2d(embedding), dim=1)
        embedding = flow.spk_embed_affine_layer(embedding)
        prompt_token = _repeat_batch_dim(prompt_token, B, ndim=2)
        prompt_token_len = _repeat_batch_dim(prompt_token_len, B, ndim=1)
        prompt_feat = _repeat_batch_dim(prompt_feat, B, ndim=3)
        prompt_feat_len = _repeat_batch_dim(prompt_feat_len, B, ndim=1)
        embedding = _repeat_batch_dim(embedding, B, ndim=2)
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = flow.input_embedding(token.long()) * mask
        h, h_masks = flow.encoder(token, token_len)
        h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
        if finalize is False:
            cut = flow.pre_lookahead_len * flow.token_mel_ratio
            h = h[:, :-cut]
            h_lengths = h_lengths - cut                      # THE FIX: upstream leaves h_lengths at the untrimmed length
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = flow.encoder_proj(h)
        conds = torch.zeros([B, mel_len1 + mel_len2, flow.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)
        mask = (~make_pad_mask(h_lengths)).unsqueeze(1).to(h)
        if mask.shape[0] != B:
            mask = mask.repeat(B, 1, 1)
        feat, _ = flow.decoder(mu=h.transpose(1, 2).contiguous(), mask=mask, spks=embedding, cond=conds, n_timesteps=n_timesteps,
                               noised_mels=noised_mels, meanflow=meanflow)
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat


def make_window_decoder(s3gen, ref_dict: dict, *, n_cfm_timesteps: Optional[int] = None):
    """A `decode(window_tokens, final) -> float32 samples @24 kHz` for `StreamDecoder`. `ref_dict` is the voice's `conds.gen`; it is shallow-copied so the
    shared voice is never mutated. final=False -> audio for len(window)-3 tokens (look-ahead withheld); final=True -> every token.
    Window = [reference prompt tokens (prepended by the flow)] + the given tokens: the already-emitted CONTEXT tokens get no mel conditioning
    (only the reference does). Whether that costs quality at window boundaries is exactly what the sweep in scripts/gpu_validate_streaming.py measures."""
    import torch
    steps = n_cfm_timesteps or (2 if s3gen.meanflow else 10)
    ref = dict(ref_dict)
    for k, v in list(ref.items()):                           # same casting S3Token2Mel.forward does, done once per request
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        if torch.is_tensor(v):
            ref[k] = v.to(device=s3gen.device, dtype=s3gen.dtype)

    def decode(window_tokens, final: bool) -> np.ndarray:
        with torch.inference_mode():
            tok = torch.tensor([list(window_tokens)], dtype=torch.long, device=s3gen.device)
            mels = flow_inference_nonfinal(s3gen.flow, token=tok, token_len=torch.LongTensor([tok.size(-1)]).to(s3gen.device),
                                           finalize=bool(final), n_timesteps=steps, meanflow=s3gen.meanflow, **ref)
            wavs, _ = s3gen.hift_inference(mels.to(dtype=s3gen.dtype), None)
            wavs[:, :len(s3gen.trim_fade)] *= s3gen.trim_fade   # as upstream: every decode fades in its first 40 ms (the context region absorbs it)
            return wavs.squeeze(0).float().cpu().numpy()
    return decode
