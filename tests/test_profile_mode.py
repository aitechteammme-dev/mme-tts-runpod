"""Profile mode of handler.py (experimental branch): off by default, and when on it only ADDS a timing_detail field. Fake model, CPU only; no torch, no GPU."""
import sys
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("soundfile")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import handler as h  # noqa: E402

BASELINE_KEYS = {"audio_base64", "format", "sample_rate", "language", "voice", "timing"}
REQ = {"input": {"text": "यह एक परीक्षण वाक्य है", "language": "hi", "output_format": "wav"}}


class FakeModel:
    """Same shape as ChatterboxMultilingualTTS for the attributes the profiler wraps; generate() calls the stages in the real order with known, tiny delays."""
    sr = 24000
    conds = {}

    class _Tok:
        def text_to_tokens(self, text, language_id=None):
            time.sleep(0.005)
            return np.zeros((1, 5), dtype=np.int64)

    class _T3:
        def inference(self, **kw):
            time.sleep(0.03)
            return np.zeros((2, 40), dtype=np.int64)             # (B, n_tokens): the wrapper counts the second dimension

    class _S3:
        def flow_inference(self, *a, **k):
            time.sleep(0.01)
            return "mels"

        def hift_inference(self, *a, **k):
            time.sleep(0.008)
            return "wav"

    class _WM:
        def apply_watermark(self, wav, sample_rate=None):
            time.sleep(0.004)
            return wav

    def __init__(self):
        self.tokenizer, self.t3, self.s3gen, self.watermarker = self._Tok(), self._T3(), self._S3(), self._WM()

    def prepare_conditionals(self, p):
        pass

    def generate(self, text, language_id="hi", **kw):
        self.tokenizer.text_to_tokens(text, language_id=language_id)
        self.t3.inference(t3_cond=None, text_tokens=None)
        self.s3gen.flow_inference(None)
        self.s3gen.hift_inference(None)
        time.sleep(0.002)                                            # "other" work inside generate()
        n = int(self.sr * 0.1 * max(1, len(text.split())))
        wav = (0.3 * np.sin(2 * np.pi * 300 * np.arange(n) / self.sr)).astype(np.float32)
        return self.watermarker.apply_watermark(wav, sample_rate=self.sr)


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(h, "MODEL", FakeModel())
    monkeypatch.setattr(h, "_torch", None)
    monkeypatch.setattr(h, "VOICES", {"utkarsh": {}})
    monkeypatch.setattr(h, "_active_voice", None)
    monkeypatch.setattr(h, "log", lambda m: None)
    h._P.enabled, h._P.d, h._P.meta = False, {}, {}
    yield
    h._P.enabled, h._P.d, h._P.meta = False, {}, {}


def test_profile_is_off_by_default_and_the_response_is_the_baseline_one():
    assert h.PROFILE is False
    r = h.handler(REQ)
    assert set(r) == BASELINE_KEYS and h._P.enabled is False


def test_profile_on_adds_one_field_and_leaves_the_audio_and_baseline_fields_unchanged():
    off = h.handler(REQ)
    assert h.install_profiling(h.MODEL, None) is True
    on = h.handler(REQ)
    assert set(on) == BASELINE_KEYS | {"timing_detail"}
    assert on["audio_base64"] == off["audio_base64"] and {k: on[k] for k in BASELINE_KEYS - {"timing"}} == {k: off[k] for k in BASELINE_KEYS - {"timing"}}


def test_every_stage_is_measured_and_the_breakdown_is_consistent():
    h.install_profiling(h.MODEL, None)
    r = h.handler(REQ)
    d = r["timing_detail"]
    for key in ("validate_s", "lock_wait_s", "select_voice_s", "tokenize_s", "t3_s", "flow_s", "hift_s", "watermark_s", "model_generate_s", "sync_and_host_copy_s", "encode_s", "model_generate_other_s", "handler_unattributed_s"):
        assert key in d and d[key] >= 0, key
    assert d["t3_s"] >= 0.03 and d["flow_s"] >= 0.01 and d["hift_s"] >= 0.008 and d["watermark_s"] >= 0.004 and d["tokenize_s"] >= 0.005
    assert d["speech_tokens"] == 40 and d["t3_ms_per_token"] == pytest.approx(1000 * d["t3_s"] / 40, rel=0.01)
    parts = d["tokenize_s"] + d["t3_s"] + d["flow_s"] + d["hift_s"] + d["watermark_s"] + d["model_generate_other_s"]
    assert parts == pytest.approx(d["model_generate_s"], abs=0.003)                                   # the stages + "other" account for the whole generate() call
    assert d["model_generate_s"] <= r["timing"]["generate_s"] + 0.01
    assert d["chars"] == len(REQ["input"]["text"]) and d["output_format"] == "wav" and d["response_base64_chars"] == len(r["audio_base64"])


def test_metadata_never_contains_credentials_or_text_or_audio(monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "worker-abc123")
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_SHOULD_NEVER_APPEAR_0123456789abcdef")
    h.install_profiling(h.MODEL, None)
    d = h.handler(REQ)["timing_detail"]
    blob = repr(d)
    assert d["worker_id"] == "worker-abc123" and "rpa_SHOULD_NEVER" not in blob and REQ["input"]["text"] not in blob and "audio_base64" not in blob


def test_profiles_do_not_accumulate_across_requests():
    h.install_profiling(h.MODEL, None)
    a = h.handler(REQ)["timing_detail"]["t3_s"]
    b = h.handler(REQ)["timing_detail"]["t3_s"]
    assert b < a + 0.05 and b < 0.2                                                                      # the second request reports its own T3 time, not the sum


def test_a_model_without_the_expected_stages_is_not_wrapped_and_still_serves():
    class Plain:
        sr = 24000
        conds = {}

        def prepare_conditionals(self, p):
            pass

        def generate(self, text, language_id="hi", **kw):
            return np.zeros(2400, dtype=np.float32)
    h.MODEL = Plain()
    assert h.install_profiling(h.MODEL, None) is False and h._P.enabled is False
    assert set(h.handler(REQ)) == BASELINE_KEYS


def test_wrappers_are_instance_attributes_and_the_pinned_class_is_untouched():
    cls_t3 = type(h.MODEL.t3)
    before = cls_t3.inference
    h.install_profiling(h.MODEL, None)
    assert "inference" in vars(h.MODEL.t3) and cls_t3.inference is before                              # the class attribute (what the fingerprints cover) is unchanged


def test_errors_inside_a_stage_still_propagate_as_handler_errors():
    h.install_profiling(h.MODEL, None)

    def boom(*a, **k):
        raise RuntimeError("stage failed")
    h.MODEL.s3gen.flow_inference = h._P and (lambda *a, **k: boom())
    r = h.handler(REQ)
    assert r.get("error") == "stage failed" and "audio_base64" not in r
