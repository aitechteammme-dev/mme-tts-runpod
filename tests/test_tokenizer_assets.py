"""The two tokenizer assets (Cangjie5_TC.json and the spacy_pkuseg model) must be local, hash-verified and never fetched at run time. CPU only.

Unit tests use synthetic files and pins. The integration tests at the bottom use the REAL assets and run only when these variables point at them (they are skipped otherwise, never faked):
  MME_TEST_MODEL_DIR     a checkpoint dir holding grapheme_mtl_merged_expanded_v1.json, Cangjie5_TC.json and the cache layout made by pinned.assets.build_cangjie_cache
  MME_TEST_PKUSEG_HOME   a folder made by `python -m pinned.fetch_pkuseg`
and the packages chatterbox, torch and spacy_pkuseg are importable.
"""
import hashlib
import io
import os
import socket
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinned import assets, pins, runtime  # noqa: E402
from pinned.errors import PinnedRuntimeError  # noqa: E402


def _sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------------------------------------- pins
def test_the_pinned_assets_are_what_the_checkpoint_pins_say():
    assert pins.CANGJIE_FILE in pins.CHECKPOINT_FILES                      # the Cangjie file is hash-pinned with the rest of the checkpoint
    assert pins.PKUSEG_URL.startswith("https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/")
    assert len(pins.PKUSEG_ARCHIVE[1]) == 64 and set(pins.PKUSEG_FILES) == {"features.msgpack", "weights.npz"}


def test_pkuseg_pin_equals_the_hash_inside_the_installed_package():
    pytest.importorskip("spacy_pkuseg")
    assets.verify_pkuseg_package()                                          # the package's own URL and model_hash equal our pins, and its version is the pinned one


# ---------------------------------------------------------------------------------------------------------- Cangjie cache
CANGJIE = b'["a\\tb"]'
CPIN = (len(CANGJIE), _sha(CANGJIE))


def _model_dir(tmp_path, data=CANGJIE):
    (tmp_path / pins.CANGJIE_FILE).write_bytes(data)
    return tmp_path


def test_cangjie_cache_layout_is_what_hf_hub_download_reads(tmp_path):
    d = _model_dir(tmp_path)
    snap = assets.build_cangjie_cache(d, pin=CPIN)
    assert snap == d / "models--ResembleAI--chatterbox" / "snapshots" / pins.CHECKPOINT_REVISION / pins.CANGJIE_FILE
    assert (d / "models--ResembleAI--chatterbox" / "refs" / "main").read_text() == pins.CHECKPOINT_REVISION
    assets.verify_cangjie_cache(d, pin=CPIN)


def test_cangjie_build_refuses_a_wrong_source_file(tmp_path):
    with pytest.raises(PinnedRuntimeError, match="sha256"):
        assets.build_cangjie_cache(_model_dir(tmp_path, b'["x\\ty"]'), pin=CPIN)           # same size, different bytes


def test_cangjie_verify_fails_clearly(tmp_path):
    d = _model_dir(tmp_path)
    with pytest.raises(PinnedRuntimeError, match="refs"):                                  # nothing built yet
        assets.verify_cangjie_cache(d, pin=CPIN)
    snap = assets.build_cangjie_cache(d, pin=CPIN)
    snap.write_bytes(b'["x\\ty"]')
    with pytest.raises(PinnedRuntimeError, match="sha256"):
        assets.verify_cangjie_cache(d, pin=CPIN)
    snap.unlink()
    with pytest.raises(PinnedRuntimeError, match="missing"):
        assets.verify_cangjie_cache(d, pin=CPIN)
    assets.build_cangjie_cache(d, pin=CPIN)
    (d / "models--ResembleAI--chatterbox" / "refs" / "main").write_text("0" * 40)
    with pytest.raises(PinnedRuntimeError, match="pinned revision"):
        assets.verify_cangjie_cache(d, pin=CPIN)


# ---------------------------------------------------------------------------------------------------------- pkuseg archive
FILES = {"features.msgpack": b"feat" * 10, "weights.npz": b"wts" * 7}
FPIN = {k: (len(v), _sha(v)) for k, v in FILES.items()}


def _zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in entries.items():
            z.writestr(k, v)
    return buf.getvalue()


def _opener(payload, calls=None):
    def opener(url, timeout=0):
        if calls is not None:
            calls.append(url)
        return io.BytesIO(payload)
    return opener


def test_fetch_pkuseg_verifies_then_installs(tmp_path):
    blob = _zip(FILES)
    calls = []
    assets.fetch_pkuseg(tmp_path / "home", url="https://example.invalid/x.zip", archive=(len(blob), _sha(blob)), files=FPIN, opener=_opener(blob, calls))
    assert calls == ["https://example.invalid/x.zip"]
    assets.verify_pkuseg(tmp_path / "home", (len(blob), _sha(blob)), FPIN)
    assert not list((tmp_path / "home").glob("*.part"))


def test_fetch_pkuseg_rejects_a_wrong_archive_and_leaves_nothing_usable(tmp_path):
    blob = _zip(FILES)
    with pytest.raises(PinnedRuntimeError, match="sha256"):
        assets.fetch_pkuseg(tmp_path / "home", archive=(len(blob), "0" * 64), files=FPIN, opener=_opener(blob))
    home = tmp_path / "home"
    assert not (home / "spacy_ontonotes.zip").exists() and not (home / "spacy_ontonotes").exists() and not list(home.glob("*.part"))


def test_fetch_pkuseg_rejects_oversize_and_wrong_entries(tmp_path):
    blob = _zip(FILES)
    with pytest.raises(PinnedRuntimeError, match="larger"):
        assets.fetch_pkuseg(tmp_path / "a", archive=(len(blob) - 1, _sha(blob)), files=FPIN, opener=_opener(blob))
    evil = _zip({**FILES, "../evil": b"x"})
    with pytest.raises(PinnedRuntimeError, match="entries"):
        assets.fetch_pkuseg(tmp_path / "b", archive=(len(evil), _sha(evil)), files=FPIN, opener=_opener(evil))
    assert not (tmp_path / "evil").exists()


def test_fetch_pkuseg_refuses_to_reuse_existing_state(tmp_path):
    blob = _zip(FILES)
    (tmp_path / "spacy_ontonotes.zip").write_bytes(b"unverified")
    with pytest.raises(PinnedRuntimeError, match="already holds"):
        assets.fetch_pkuseg(tmp_path, archive=(len(blob), _sha(blob)), files=FPIN, opener=_opener(blob))


def test_verify_pkuseg_fails_clearly(tmp_path):
    blob = _zip(FILES)
    arch = (len(blob), _sha(blob))
    with pytest.raises(PinnedRuntimeError, match="missing"):
        assets.verify_pkuseg(tmp_path, arch, FPIN)                                          # no zip: spacy_pkuseg would DOWNLOAD
    assets.fetch_pkuseg(tmp_path, archive=arch, files=FPIN, opener=_opener(blob))
    m = tmp_path / "spacy_ontonotes"
    (m / "weights.npz").write_bytes(b"tampered")
    with pytest.raises(PinnedRuntimeError, match="weights.npz"):
        assets.verify_pkuseg(tmp_path, arch, FPIN)
    (m / "weights.npz").write_bytes(FILES["weights.npz"])
    (m / "extra.bin").write_bytes(b"x")
    with pytest.raises(PinnedRuntimeError, match="unexpected"):
        assets.verify_pkuseg(tmp_path, arch, FPIN)
    (m / "extra.bin").unlink()
    (m / "features.msgpack").unlink()
    with pytest.raises(PinnedRuntimeError, match="features.msgpack"):
        assets.verify_pkuseg(tmp_path, arch, FPIN)
    (tmp_path / "spacy_ontonotes.zip").write_bytes(blob + b"0")
    with pytest.raises(PinnedRuntimeError, match="size"):
        assets.verify_pkuseg(tmp_path, arch, FPIN)


# ---------------------------------------------------------------------------------------------------------- network denial and environment
def test_no_network_blocks_connect_and_dns_and_catches_swallowed_errors():
    with pytest.raises(PinnedRuntimeError, match="network access was attempted"):
        with runtime.no_network():
            for target in (lambda: socket.create_connection(("127.0.0.1", 9), timeout=1), lambda: socket.getaddrinfo("example.com", 443)):
                try:
                    target()
                except Exception:       # the Cangjie loader does exactly this
                    pass
    assert socket.getaddrinfo("localhost", 80)          # name resolution works again after the block


def test_no_network_allows_local_unix_sockets_and_plain_code():
    with runtime.no_network():
        a, b = socket.socketpair()
        a.close()
        b.close()


def test_tokenizer_assets_need_hf_offline_and_pkuseg_home(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import constants
    monkeypatch.setattr(assets, "verify_cangjie_cache", lambda *a, **k: None)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    with pytest.raises(PinnedRuntimeError, match="HF_HUB_OFFLINE"):
        runtime.verify_tokenizer_assets("/m", {"PKUSEG_HOME": "/p"})
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(PinnedRuntimeError, match="PKUSEG_HOME must be set"):
        runtime.verify_tokenizer_assets("/m", {})


def test_tokenizer_assets_refuse_a_pkuseg_home_the_package_does_not_use(monkeypatch):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("spacy_pkuseg")
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(assets, "verify_cangjie_cache", lambda *a, **k: None)
    monkeypatch.setattr(assets, "verify_pkuseg_package", lambda: None)
    import importlib
    monkeypatch.setattr(importlib.import_module("spacy_pkuseg.config").config, "pkuseg_home", "/somewhere/else")
    with pytest.raises(PinnedRuntimeError, match="before the first import"):
        runtime.verify_tokenizer_assets("/m", {"PKUSEG_HOME": "/app/pkuseg"})


def test_dockerfile_bakes_both_assets_before_going_offline_and_proves_it():
    text = (ROOT / "Dockerfile").read_text()
    for needle in ("pinned.fetch_checkpoint /app/model", "pinned.fetch_pkuseg /app/pkuseg", "ENV PKUSEG_HOME=/app/pkuseg", "ENV HF_HUB_OFFLINE=1", "verify_tokenizer_offline"):
        assert needle in text, needle
    assert text.index("fetch_pkuseg") < text.index("ENV HF_HUB_OFFLINE=1") < text.index("verify_tokenizer_offline")   # downloads (hash-checked) first, then offline, then the proof
    assert "snapshot_download('ResembleAI/chatterbox')" not in text


def test_handler_sets_the_offline_environment_before_importing_anything_that_reads_it():
    src = (ROOT / "handler.py").read_text(encoding="utf-8")
    assert src.index('os.environ.setdefault("HF_HUB_OFFLINE"') < src.index("from pinned import")
    assert src.index('os.environ.setdefault("PKUSEG_HOME"') < src.index("from pinned import")


# ---------------------------------------------------------------------------------------------------------- integration with the REAL assets
MODEL_DIR = os.environ.get("MME_TEST_MODEL_DIR")
PKUSEG_HOME = os.environ.get("MME_TEST_PKUSEG_HOME")
needs_assets = pytest.mark.skipif(not (MODEL_DIR and PKUSEG_HOME), reason="MME_TEST_MODEL_DIR / MME_TEST_PKUSEG_HOME not set")

SCRIPT = r"""
import sys, json
sys.path.insert(0, {root!r})
from pinned import runtime, pins
out = {{}}
out["words"] = runtime.verify_tokenizer_offline({model!r})
from chatterbox.models.tokenizers import MTLTokenizer
with runtime.no_network():
    tok = MTLTokenizer({model!r} + "/grapheme_mtl_merged_expanded_v1.json")
conv = tok.cangjie_converter
text = "你好,今天天气很好"
out["converted"] = conv(text)
out["segmented"] = list(conv.segmenter.cut(text))
saved = conv.word2cj
conv.word2cj = {{}}
out["converted_without_mapping"] = conv(text)
conv.word2cj = saved
out["tokens_zh"] = tok.encode(text, language_id="zh")[:12]
print("RESULT" + json.dumps(out, ensure_ascii=False))
"""


def _run(model_dir, home, extra_env=None):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PKUSEG_HOME": home, **(extra_env or {})}
    return subprocess.run([sys.executable, "-c", SCRIPT.format(root=str(ROOT), model=model_dir)], env=env, capture_output=True, text=True, timeout=600)


@needs_assets
def test_real_assets_verify_and_initialize_offline():
    pytest.importorskip("torch")
    pytest.importorskip("chatterbox")
    pytest.importorskip("spacy_pkuseg")
    assets.verify_pkuseg(PKUSEG_HOME)
    assets.verify_cangjie_cache(MODEL_DIR)
    r = _run(MODEL_DIR, PKUSEG_HOME)
    assert r.returncode == 0, r.stderr[-1500:]
    import json
    out = json.loads(r.stdout.split("RESULT", 1)[1])
    assert out["words"] == pins.CANGJIE_DISTINCT_WORDS                                      # the pinned Cangjie file was actually loaded
    assert len(out["segmented"]) >= 2 and "".join(out["segmented"]) == "你好,今天天气很好"        # the pinned local pkuseg model initialized and segments
    assert "[cj_" in out["converted"] and "[cj_" not in out["converted_without_mapping"]    # and the mapping is USED by Chinese tokenization
    assert out["tokens_zh"]


@needs_assets
def test_real_run_fails_clearly_when_the_pkuseg_model_is_missing(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("chatterbox")
    r = _run(MODEL_DIR, str(tmp_path))
    assert r.returncode != 0 and "pkuseg model archive is missing" in r.stderr           # a clear error, and no download was attempted: the folder is still empty
    assert not list(tmp_path.iterdir())


@needs_assets
def test_real_run_fails_clearly_when_the_cangjie_cache_is_corrupt(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("chatterbox")
    import shutil
    copy = tmp_path / "model"
    shutil.copytree(MODEL_DIR, copy)
    assets.cangjie_snapshot_file(copy).write_bytes(b"[]")
    r = _run(str(copy), PKUSEG_HOME)
    assert r.returncode != 0 and "Cangjie cache file" in r.stderr
