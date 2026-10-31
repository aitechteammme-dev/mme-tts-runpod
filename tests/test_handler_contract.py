"""Handler contract tests with a fake model (no torch, no chatterbox, no GPU). Skipped when numpy / scipy / soundfile are missing."""
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")
pytest.importorskip("soundfile")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import handler as h  # noqa: E402


class FakeModel:
    sr = 24000
    conds = {}

    def prepare_conditionals(self, path):
        pass

    def generate(self, text, language_id="hi", **kw):
        return np.zeros(2400, dtype=np.float32)


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    monkeypatch.setattr(h, "MODEL", FakeModel())
    monkeypatch.setattr(h, "_torch", None)
    monkeypatch.setattr(h, "VOICES", {"utkarsh": {}})
    monkeypatch.setattr(h, "log", lambda m: None)


BASE = {"text": "नमस्ते", "language": "hi"}


@pytest.mark.parametrize("flag", [True, 1, 2, "true", "false", "0", "no", [1], {"a": 1}])
def test_every_truthy_stream_value_is_refused_without_audio(flag):
    r = h.handler({"input": {**BASE, "stream": flag}})
    assert r["error"] == "streaming_disabled" and "audio_base64" not in r


@pytest.mark.parametrize("flag", [False, 0, None, "", [], {}])
def test_falsy_stream_values_return_the_full_waveform(flag):
    r = h.handler({"input": {**BASE, "stream": flag}})
    assert "audio_base64" in r and "error" not in r


def test_stream_is_refused_before_any_other_validation():
    assert h.handler({"input": {"stream": True}})["error"] == "streaming_disabled"      # not "Provide a non-empty input.text"


def test_refusal_does_not_touch_the_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("generate must not run for a refused stream request")
    monkeypatch.setattr(h.MODEL, "generate", boom)
    assert h.handler({"input": {**BASE, "stream": True}})["error"] == "streaming_disabled"


@pytest.mark.parametrize("job", [{}, {"input": None}, {"input": []}, {"input": {}}])
def test_missing_input_is_an_error_not_a_crash(job):
    assert "error" in h.handler(job)


def test_streaming_code_path_exists_in_the_handler():
    src = (ROOT / "handler.py").read_text(encoding="utf-8")
    assert re.search(r"^\s*yield\b", src, re.M) and "return_aggregate_stream" in src


def test_lock_pins_are_exact_and_cover_requirements_txt():
    lock = {}
    for line in (ROOT / "requirements.lock").read_text().splitlines():
        m = re.match(r"([A-Za-z0-9_.\-]+)==([^\s\\]+)", line)               # "name==1.2.3 \" followed by indented --hash lines
        if m:
            lock[m.group(1).lower().replace("_", "-")] = m.group(2)
    assert not {"torch", "torchaudio"} & set(lock)                       # installed from the cu128 index by the Dockerfile
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"([A-Za-z0-9_.\-]+)==(\S+)$", line.strip())
        if m:
            assert lock[m.group(1).lower().replace("_", "-")] == m.group(2), m.group(1)
