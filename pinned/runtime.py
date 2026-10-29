"""Fail-closed start-up checks for the v4 worker. Every function raises PinnedRuntimeError on the first mismatch: there is no warning-and-continue path, and nothing falls back to V2.

  verify_checkpoint(dir)            size + sha256 of every required file, and no second T3 file next to the V3 one
  verify_distributions()            chatterbox-tts and resemble-perth installed from the pinned commits; torch / torchaudio / transformers at the pinned versions
  verify_source_fingerprints()      the installed Chatterbox functions are the ones the adapter and this worker were reviewed against
  load_verified_model(device, dir)  all of the above, then from_local with the explicit V3 file, then proof that the watermarker and generate() are the real ones
  streaming_refusal()               the error that replaces any streamed audio while STREAMING_ALLOWED is False
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import socket
from importlib import metadata
from pathlib import Path

from . import assets, fingerprint, pins
from .errors import PinnedRuntimeError   # noqa: F401  (re-exported: callers use runtime.PinnedRuntimeError)


def _fail(msg: str):
    raise PinnedRuntimeError(msg)


# ---------------------------------------------------------------------------------------------------------- checkpoint
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def verify_checkpoint(model_dir, files=None, hasher=sha256_file) -> None:
    """Every required file exists with the pinned size and sha256. A V2 T3 file (or any other *.safetensors named t3_*) in the directory is refused too: the V3 file must be the only T3 there."""
    files = pins.CHECKPOINT_FILES if files is None else files
    d = Path(model_dir)
    if not d.is_dir():
        _fail(f"checkpoint directory {d} does not exist")
    for name, (size, digest) in files.items():
        p = d / name
        if not p.is_file():
            _fail(f"required checkpoint file missing: {p}")
        if p.stat().st_size != size:
            _fail(f"{name}: size {p.stat().st_size} != pinned {size}")
        got = hasher(p)
        if got != digest:
            _fail(f"{name}: sha256 {got} != pinned {digest} (checkpoint revision {pins.CHECKPOINT_REVISION})")
    other_t3 = sorted(p.name for p in d.glob("t3_*") if p.name != pins.T3_FILE)
    if other_t3:
        _fail(f"unexpected T3 file(s) in {d}: {other_t3}; only {pins.T3_FILE} may be present (no V2 fallback)")


# ---------------------------------------------------------------------------------------------------------- installed packages
def _direct_url_commit(dist_name: str) -> str | None:
    try:
        raw = metadata.distribution(dist_name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        _fail(f"{dist_name} is not installed")
    if not raw:
        return None
    return (json.loads(raw).get("vcs_info") or {}).get("commit_id")


def _base_version(v: str) -> str:
    return v.split("+")[0]      # torch 2.7.1+cu128 -> 2.7.1


def verify_distributions() -> None:
    for dist, commit, version in ((pins.CHATTERBOX_DIST, pins.CHATTERBOX_COMMIT, pins.CHATTERBOX_VERSION), (pins.PERTH_DIST, pins.PERTH_COMMIT, pins.PERTH_VERSION)):
        got = _direct_url_commit(dist)
        if got != commit:
            _fail(f"{dist} must be installed from git commit {commit}; installed commit is {got!r}")
        if metadata.version(dist) != version:
            _fail(f"{dist} version {metadata.version(dist)} != pinned {version}")
    for dist, want in (("torch", pins.TORCH_VERSION), ("torchaudio", pins.TORCHAUDIO_VERSION), ("transformers", pins.TRANSFORMERS_VERSION)):
        try:
            got = metadata.version(dist)
        except metadata.PackageNotFoundError:
            _fail(f"{dist} is not installed")
        if _base_version(got) != want:
            _fail(f"{dist} {got} != pinned {want}")


def _package_dir() -> Path:
    spec = importlib.util.find_spec("chatterbox")        # does not import the package
    if spec is None or not spec.submodule_search_locations:
        _fail("chatterbox is not installed")
    return Path(list(spec.submodule_search_locations)[0])


def verify_source_fingerprints(package_dir=None, table=None) -> None:
    root = Path(package_dir) if package_dir else _package_dir()
    table = pins.SOURCE_FINGERPRINTS if table is None else table
    cache: dict[str, str] = {}
    bad = []
    for qual, (rel, want) in table.items():
        try:
            src = cache.setdefault(rel, (root / rel).read_text(encoding="utf-8"))
            node, _ = fingerprint.find_function(src, qual)
            got = fingerprint.canonical_fingerprint(node)
        except (OSError, KeyError, SyntaxError) as exc:
            bad.append(f"{qual}: cannot be fingerprinted ({type(exc).__name__}: {exc})")
            continue
        if got != want:
            bad.append(f"{qual}: {got[:16]}... != pinned {want[:16]}...")
    if bad:
        _fail("installed Chatterbox source differs from commit " + pins.CHATTERBOX_COMMIT + ": " + "; ".join(bad))


# ---------------------------------------------------------------------------------------------------------- tokenizer assets and the network
def verify_tokenizer_assets(model_dir, env=None) -> Path:
    """Everything the pinned tokenizer needs is local, verified and will be found WITHOUT a request: the Cangjie cache layout inside the checkpoint dir, the pkuseg model in PKUSEG_HOME,
    huggingface_hub in offline mode. Returns the pkuseg home."""
    env = os.environ if env is None else env
    assets.verify_cangjie_cache(model_dir)
    try:
        from huggingface_hub import constants
    except ImportError as exc:
        _fail(f"huggingface_hub is required: {exc}")
    if not constants.HF_HUB_OFFLINE:
        _fail("HF_HUB_OFFLINE must be 1 before huggingface_hub is imported: the tokenizer's hf_hub_download call would otherwise reach the Hub")
    home = env.get("PKUSEG_HOME")
    if not home:
        _fail("PKUSEG_HOME must be set (to the verified pkuseg folder) before spacy_pkuseg is imported: it would otherwise use ~/.pkuseg and download its model")
    assets.verify_pkuseg_package()
    import importlib
    seen = importlib.import_module("spacy_pkuseg.config").config.pkuseg_home
    if os.path.realpath(os.path.expanduser(home)) != os.path.realpath(seen):
        _fail(f"spacy_pkuseg was imported with pkuseg_home={seen!r} but PKUSEG_HOME={home!r}: set the variable before the first import")
    assets.verify_pkuseg(seen)
    return Path(seen)


@contextlib.contextmanager
def no_network():
    """Inside the block every INET connection attempt and DNS lookup is refused and recorded. Code that catches the error and carries on (the Cangjie loader does) is still caught when the block exits."""
    attempts: list[str] = []
    orig = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)

    def _is_net(sock):
        return sock.family in (socket.AF_INET, socket.AF_INET6)

    def connect(self, address):
        if _is_net(self):
            attempts.append(f"connect {address!r}")
            raise PinnedRuntimeError(f"network connection attempted during pinned initialization: {address!r}")
        return orig[0](self, address)

    def connect_ex(self, address):
        if _is_net(self):
            attempts.append(f"connect_ex {address!r}")
            raise PinnedRuntimeError(f"network connection attempted during pinned initialization: {address!r}")
        return orig[1](self, address)

    def getaddrinfo(host, *a, **k):
        attempts.append(f"dns {host!r}")
        raise PinnedRuntimeError(f"DNS lookup attempted during pinned initialization: {host!r}")

    socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo
    try:
        yield attempts
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = orig
    if attempts:
        _fail("network access was attempted during initialization (" + "; ".join(attempts) + "): an asset is missing from the image")


def check_loaded_tokenizer(tokenizer) -> None:
    """The Cangjie mapping really came from the pinned file and the pkuseg segmenter really initialized: otherwise Chinese would degrade with only a log warning."""
    conv = getattr(tokenizer, "cangjie_converter", None)
    if conv is None:
        _fail("the tokenizer has no cangjie_converter")
    n = len(getattr(conv, "word2cj", {}) or {})
    if n != pins.CANGJIE_DISTINCT_WORDS:
        _fail(f"Cangjie mapping holds {n} words, expected {pins.CANGJIE_DISTINCT_WORDS} from the pinned {pins.CANGJIE_FILE}: Chinese tokenization would be degraded")
    if getattr(conv, "segmenter", None) is None:
        _fail("the pkuseg segmenter is not initialized: Chinese segmentation would be degraded")


def verify_tokenizer_offline(model_dir, env=None) -> int:
    """Build the tokenizer exactly as from_local does, with the network denied, and check it. Image-build smoke test (needs chatterbox and torch, no GPU). Returns the Cangjie word count."""
    verify_tokenizer_assets(model_dir, env)
    from chatterbox.models.tokenizers import MTLTokenizer
    with no_network():
        tok = MTLTokenizer(str(Path(model_dir) / "grapheme_mtl_merged_expanded_v1.json"))
    check_loaded_tokenizer(tok)
    return len(tok.cangjie_converter.word2cj)


# ---------------------------------------------------------------------------------------------------------- model
def load_verified_model(device: str, model_dir, env=None, log=print):
    """Load ChatterboxMultilingualTTS from the verified local checkpoint with the explicit V3 T3 file. Raises PinnedRuntimeError instead of falling back to anything."""
    env = os.environ if env is None else env
    asked = env.get("T3_MODEL")
    if asked not in (None, "", "v3", pins.T3_FILE):
        _fail(f"T3_MODEL={asked!r} is refused: v4 loads only {pins.T3_FILE}")
    verify_distributions()
    verify_source_fingerprints()
    verify_checkpoint(model_dir)
    verify_tokenizer_assets(model_dir, env)
    import inspect
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    if "t3_model" not in inspect.signature(ChatterboxMultilingualTTS.from_local).parameters:
        _fail("this Chatterbox from_local has no t3_model parameter: it would load the library default (V2)")
    with no_network():       # from_local must need nothing from the network: any attempt (even a swallowed one) fails the start-up
        model = ChatterboxMultilingualTTS.from_local(Path(model_dir), device, t3_model=pins.T3_FILE)
    verify_loaded_model(model)
    log(f"[TTS] verified: chatterbox {pins.CHATTERBOX_COMMIT[:12]}, checkpoint {pins.CHECKPOINT_REVISION[:12]}, t3={pins.T3_FILE}, s3gen={pins.S3GEN_FILE}")
    return model


def verify_loaded_model(model) -> None:
    """The watermarked full-waveform path must be intact: the real Perth watermarker, and the unmodified generate() (which applies it to the whole waveform)."""
    try:
        import perth
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as exc:
        _fail(f"cannot verify the watermark path: {exc}")
    if type(model).generate is not ChatterboxMultilingualTTS.generate:
        _fail("the model class overrides generate(): the watermarked path is not the verified one")
    if not isinstance(getattr(model, "watermarker", None), perth.PerthImplicitWatermarker):
        _fail("the model has no PerthImplicitWatermarker: audio would not be watermarked")
    check_loaded_tokenizer(getattr(model, "tokenizer", None))


# ---------------------------------------------------------------------------------------------------------- streaming
def streaming_refusal() -> dict | None:
    """None when streamed audio may be emitted; otherwise the error to return INSTEAD of any audio."""
    if pins.STREAMING_ALLOWED:
        return None
    return {"error": "streaming_disabled", "detail": "streamed audio would not carry the Perth watermark; no verified chunked watermark method exists. Request the full waveform (omit input.stream)."}
