"""Offline tests of the fail-closed pinned runtime. No torch, no chatterbox, no GPU, no network. Run: python -m pytest tests -p no:cacheprovider"""
import hashlib
import sys
import textwrap
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinned import fingerprint, pins, runtime  # noqa: E402


def _table(tmp_path, contents):
    files = {}
    for name, data in contents.items():
        (tmp_path / name).write_bytes(data)
        files[name] = (len(data), hashlib.sha256(data).hexdigest())
    return files


# ---------------------------------------------------------------------------------------------------------- pins
def test_pins_are_exact():
    assert len(pins.CHATTERBOX_COMMIT) == 40 and len(pins.CHECKPOINT_REVISION) == 40 and len(pins.PERTH_COMMIT) == 40
    assert pins.T3_FILE in pins.CHECKPOINT_FILES and pins.S3GEN_FILE in pins.CHECKPOINT_FILES
    assert pins.V2_T3_FILE not in pins.CHECKPOINT_FILES
    assert all(len(h) == 64 for _, h in pins.CHECKPOINT_FILES.values())
    assert pins.STREAMING_ALLOWED is False


def test_dockerfile_commits_match_pins():
    text = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()
    assert pins.CHATTERBOX_COMMIT in text and pins.PERTH_COMMIT in text
    assert "snapshot_download('ResembleAI/chatterbox')" not in text      # the unpinned download is gone


# ---------------------------------------------------------------------------------------------------------- checkpoint
def test_checkpoint_ok(tmp_path):
    runtime.verify_checkpoint(tmp_path, _table(tmp_path, {"a.bin": b"aaa", "b.bin": b"bbbb"}))


def test_checkpoint_missing_file(tmp_path):
    files = _table(tmp_path, {"a.bin": b"aaa"})
    files["gone.bin"] = (1, "0" * 64)
    with pytest.raises(runtime.PinnedRuntimeError, match="missing"):
        runtime.verify_checkpoint(tmp_path, files)


def test_checkpoint_wrong_size_and_wrong_hash(tmp_path):
    files = _table(tmp_path, {"a.bin": b"aaa"})
    (tmp_path / "a.bin").write_bytes(b"aaaa")
    with pytest.raises(runtime.PinnedRuntimeError, match="size"):
        runtime.verify_checkpoint(tmp_path, files)
    (tmp_path / "a.bin").write_bytes(b"aab")                              # same size, different bytes
    with pytest.raises(runtime.PinnedRuntimeError, match="sha256"):
        runtime.verify_checkpoint(tmp_path, files)


def test_checkpoint_refuses_a_v2_t3_file_next_to_v3(tmp_path):
    files = _table(tmp_path, {pins.T3_FILE: b"v3"})
    (tmp_path / pins.V2_T3_FILE).write_bytes(b"v2")
    with pytest.raises(runtime.PinnedRuntimeError, match="no V2 fallback"):
        runtime.verify_checkpoint(tmp_path, files)


def test_checkpoint_directory_missing(tmp_path):
    with pytest.raises(runtime.PinnedRuntimeError, match="does not exist"):
        runtime.verify_checkpoint(tmp_path / "nope", {})


# ---------------------------------------------------------------------------------------------------------- fingerprints
SRC = textwrap.dedent('''
    class A:
        @staticmethod
        def f(x):
            """doc
               indented second line"""
            return x + 1   # comment

    def g(y):
        return y
''')


def _fp(src, q):
    return fingerprint.canonical_fingerprint(fingerprint.find_function(src, q)[0])


def test_fingerprint_ignores_formatting_and_comments_but_not_code():
    base = _fp(SRC, "A.f")
    assert _fp(SRC.replace("# comment", "# other comment").replace("x + 1", "x+1"), "A.f") == base
    assert _fp(SRC.replace("x + 1", "x + 2"), "A.f") != base
    assert _fp(SRC.replace("@staticmethod", ""), "A.f") != base           # decorators are part of the node


def test_fingerprint_hashes_the_node_from_the_module_not_a_dedented_reparse():
    # an indented multi-line docstring changes if the method text is dedented and re-parsed: the fingerprint must not do that
    node, text = fingerprint.find_function(SRC, "A.f")
    assert fingerprint.canonical_fingerprint(node) == _fp(SRC, "A.f")
    assert "indented second line" in text


def _package(tmp_path, source):
    (tmp_path / "mtl_tts.py").write_text(source, encoding="utf-8")
    return tmp_path


def test_source_fingerprints_match_and_mismatch(tmp_path):
    table = {"A.f": ("mtl_tts.py", _fp(SRC, "A.f")), "g": ("mtl_tts.py", _fp(SRC, "g"))}
    runtime.verify_source_fingerprints(_package(tmp_path, SRC), table)
    with pytest.raises(runtime.PinnedRuntimeError, match="A.f"):
        runtime.verify_source_fingerprints(_package(tmp_path, SRC.replace("x + 1", "x + 3")), table)
    with pytest.raises(runtime.PinnedRuntimeError, match="cannot be fingerprinted"):
        runtime.verify_source_fingerprints(_package(tmp_path, "def other(): pass"), table)


# ---------------------------------------------------------------------------------------------------------- distributions
def test_distribution_commit_mismatch_fails(monkeypatch):
    monkeypatch.setattr(runtime, "_direct_url_commit", lambda dist: "0" * 40)
    with pytest.raises(runtime.PinnedRuntimeError, match="must be installed from git commit"):
        runtime.verify_distributions()


def test_base_version_ignores_local_suffix():
    assert runtime._base_version("2.7.1+cu128") == "2.7.1"


# ---------------------------------------------------------------------------------------------------------- loading
def good_tokenizer(words=None, segmenter=True):
    conv = types.SimpleNamespace(word2cj=dict.fromkeys(range(pins.CANGJIE_DISTINCT_WORDS if words is None else words), "a"), segmenter=object() if segmenter else None)
    return types.SimpleNamespace(cangjie_converter=conv)


def _fake_chatterbox(monkeypatch, from_local, generate=None, watermarker=True, tokenizer=None):
    class Model:
        def __init__(self):
            self.watermarker = _Perth() if watermarker else None
            self.tokenizer = tokenizer if tokenizer is not None else good_tokenizer()
    Model.from_local = classmethod(from_local)
    Model.generate = generate or (lambda self, *a, **k: None)
    perth = types.ModuleType("perth")

    class _Perth:
        pass
    perth.PerthImplicitWatermarker = _Perth
    mtl = types.ModuleType("chatterbox.mtl_tts")
    mtl.ChatterboxMultilingualTTS = Model
    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mtl)
    monkeypatch.setitem(sys.modules, "perth", perth)
    for name in ("verify_distributions", "verify_source_fingerprints", "verify_checkpoint", "verify_tokenizer_assets"):
        monkeypatch.setattr(runtime, name, lambda *a, **k: None)
    return Model


def test_loader_passes_the_explicit_v3_file(monkeypatch):
    seen = {}

    def from_local(cls, ckpt_dir, device, t3_model=None):
        seen.update(ckpt_dir=ckpt_dir, device=device, t3_model=t3_model)
        return cls()
    _fake_chatterbox(monkeypatch, from_local)
    runtime.load_verified_model("cuda", "/m", {}, lambda m: None)
    assert seen["t3_model"] == pins.T3_FILE and seen["device"] == "cuda"


def test_loader_fails_closed_when_from_local_has_no_t3_model_parameter(monkeypatch):
    def from_local(cls, ckpt_dir, device):
        raise AssertionError("must not be called")
    _fake_chatterbox(monkeypatch, from_local)
    with pytest.raises(runtime.PinnedRuntimeError, match="no t3_model parameter"):
        runtime.load_verified_model("cuda", "/m", {}, lambda m: None)


@pytest.mark.parametrize("value", ["v2", "t3_mtl23ls_v2", pins.V2_T3_FILE, "anything.safetensors"])
def test_loader_refuses_other_t3_models(monkeypatch, value):
    _fake_chatterbox(monkeypatch, lambda cls, d, dev, t3_model=None: cls())
    with pytest.raises(runtime.PinnedRuntimeError, match="refused"):
        runtime.load_verified_model("cuda", "/m", {"T3_MODEL": value}, lambda m: None)


def test_loader_refuses_a_model_without_the_real_watermarker(monkeypatch):
    _fake_chatterbox(monkeypatch, lambda cls, d, dev, t3_model=None: cls(), watermarker=False)
    with pytest.raises(runtime.PinnedRuntimeError, match="Perth"):
        runtime.load_verified_model("cuda", "/m", {}, lambda m: None)


@pytest.mark.parametrize("tok, match", [(good_tokenizer(words=0), "Cangjie mapping holds 0 words"), (good_tokenizer(segmenter=False), "pkuseg segmenter"),
                                        (types.SimpleNamespace(), "no cangjie_converter")])
def test_loader_refuses_a_degraded_tokenizer(monkeypatch, tok, match):
    _fake_chatterbox(monkeypatch, lambda cls, d, dev, t3_model=None: cls(), tokenizer=tok)
    with pytest.raises(runtime.PinnedRuntimeError, match=match):
        runtime.load_verified_model("cuda", "/m", {}, lambda m: None)


def test_loader_runs_from_local_with_the_network_denied(monkeypatch):
    import socket

    def from_local(cls, ckpt_dir, device, t3_model=None):
        try:
            socket.create_connection(("huggingface.co", 443), timeout=1)      # what an unpinned hf_hub_download would do; the library swallows the error
        except Exception:
            pass
        return cls()
    _fake_chatterbox(monkeypatch, from_local)
    with pytest.raises(runtime.PinnedRuntimeError, match="network access was attempted"):
        runtime.load_verified_model("cuda", "/m", {}, lambda m: None)


def test_loader_refuses_a_subclass_that_overrides_generate(monkeypatch):
    Base = _fake_chatterbox(monkeypatch, lambda cls, d, dev, t3_model=None: cls())

    class Sub(Base):
        def generate(self, *a, **k):
            return "unwatermarked"
    with pytest.raises(runtime.PinnedRuntimeError, match="overrides generate"):
        runtime.verify_loaded_model(Sub())


# ---------------------------------------------------------------------------------------------------------- streaming
def test_streaming_is_refused_with_no_audio():
    r = runtime.streaming_refusal()
    assert r["error"] == "streaming_disabled" and "audio_base64" not in r


def test_streaming_cannot_be_enabled_by_the_environment(monkeypatch):
    for k in ("STREAMING_ALLOWED", "ALLOW_STREAMING", "STREAM", "STREAMING"):
        monkeypatch.setenv(k, "1")
    assert runtime.streaming_refusal() is not None
