"""Language-specific speaking pace (handler.py): pitch-preserving time-stretch of the finished waveform. Offline: generated audio and a fake model, no torch, no GPU, no provider request."""
import base64
import io
import sys
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import handler as h  # noqa: E402

SR = 24000
BASELINE_KEYS = {"audio_base64", "format", "sample_rate", "language", "voice", "timing"}


def speechlike(seconds=3.0, f0=140.0, sr=SR, seed=1):
    """A voiced, harmonic signal with a syllable-rate amplitude envelope and short pauses: speech-shaped enough to exercise the aligner, deterministic."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2
    env[int(0.45 * n):int(0.52 * n)] = 0.0           # a pause
    y = y * env
    return (0.4 * y / np.max(np.abs(y))).astype(np.float32)


def f0_of(x, sr=SR, lo=70.0, hi=400.0):
    """Median fundamental over voiced 50 ms windows by autocorrelation: independent of the stretch implementation."""
    win, step = int(0.05 * sr), int(0.025 * sr)
    out = []
    for s in range(0, len(x) - win, step):
        w = x[s:s + win].astype(np.float64)
        if np.sqrt(np.mean(w ** 2)) < 0.03:
            continue
        w = w - w.mean()
        ac = np.correlate(w, w, "full")[win - 1:]
        a, b = int(sr / hi), int(sr / lo)
        out.append(sr / (a + int(np.argmax(ac[a:b]))))
    return float(np.median(out))


class FakeModel:
    sr = SR
    conds = {}

    def __init__(self, wave):
        self.wave = wave

    def prepare_conditionals(self, path):
        pass

    def generate(self, text, language_id="hi", **kw):
        return self.wave.copy()


@pytest.fixture
def worker(monkeypatch):
    wave = speechlike()
    monkeypatch.setattr(h, "MODEL", FakeModel(wave))
    monkeypatch.setattr(h, "_torch", None)
    monkeypatch.setattr(h, "VOICES", {"utkarsh": {}})
    monkeypatch.setattr(h, "log", lambda m: None)
    monkeypatch.setattr(h, "PACE", {"en": 0.85, "hi": 1.0, "other": 1.0})
    return wave


def run(language, fmt="wav", **extra):
    r = h.handler({"input": {"text": "this is a test sentence", "language": language, "output_format": fmt, **extra}})
    assert "error" not in r, r
    return r


def decode(r):
    data, sr = sf.read(io.BytesIO(base64.b64decode(r["audio_base64"])), dtype="float32")
    return data, sr


# ---------------------------------------------------------------------------------------------------------- 1. Hindi unchanged
def test_hindi_output_is_byte_identical_to_the_model_output(worker, monkeypatch):
    r = run("hi")
    expected = io.BytesIO()
    sf.write(expected, worker, SR, format="WAV", subtype="PCM_16")
    assert r["audio_base64"] == base64.b64encode(expected.getvalue()).decode("ascii")


def test_hindi_is_not_stretched_even_when_english_is_slowed(worker):
    h_audio, _ = decode(run("hi"))
    assert len(h_audio) == len(worker)


def test_hinglish_and_hindi_names_map_to_hindi_and_are_not_slowed():
    for tag in ("hi", "HI", "hindi", "hinglish", "hi-IN", "hi_in"):
        assert h.pace_language(tag) == "hi"
        assert h.pace_speed(tag, {"en": 0.85, "hi": 1.0, "other": 1.0}) == 1.0


# ---------------------------------------------------------------------------------------------------------- 2. duration, 3. pitch, 4. sample rate
@pytest.mark.parametrize("speed", [0.85, 0.7, 0.9])
def test_english_duration_grows_as_expected(speed):
    x = speechlike(4.0)
    y = h.time_stretch(x, SR, speed)
    assert abs(len(y) / len(x) - 1 / speed) < 0.01


def test_english_request_is_longer_by_about_the_configured_factor(worker):
    en, _ = decode(run("en"))
    assert abs(len(en) / len(worker) - 1 / 0.85) < 0.01


def test_pitch_is_preserved():
    x = speechlike(4.0, f0=140.0)
    y = h.time_stretch(x, SR, 0.85)
    before, after = f0_of(x), f0_of(y)
    assert abs(after - before) / before < 0.03          # a resample-based slowdown would drop it by 15 %


def test_pitch_is_preserved_for_a_different_voice_range():
    x = speechlike(4.0, f0=210.0)
    y = h.time_stretch(x, SR, 0.85)
    assert abs(f0_of(y) - f0_of(x)) / f0_of(x) < 0.03


def test_level_is_preserved():
    x = speechlike(4.0)
    y = h.time_stretch(x, SR, 0.85)
    assert abs(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2)) - 1) < 0.08
    assert np.max(np.abs(y)) <= 1.0 and y.dtype == np.float32


def test_sample_rate_is_unchanged_in_every_format(worker):
    r = run("en", "wav")
    assert r["sample_rate"] == SR and decode(r)[1] == SR
    assert run("en", "pcm16")["sample_rate"] == SR
    assert run("en", "ulaw8k")["sample_rate"] == 8000


def test_ulaw_length_follows_the_stretched_audio(worker):
    n = len(base64.b64decode(run("en", "ulaw8k")["audio_base64"]))
    assert abs(n / (len(worker) / 3) - 1 / 0.85) < 0.01


# ---------------------------------------------------------------------------------------------------------- 5. silence, empty, short, hostile input
def test_empty_audio_is_returned_untouched():
    e = np.zeros(0, dtype=np.float32)
    assert h.time_stretch(e, SR, 0.85) is e


def test_very_short_clips_are_returned_unchanged_and_do_not_crash():
    for n in (1, 10, 700, 2000):
        x = (0.3 * np.sin(np.arange(n) / 5)).astype(np.float32)
        assert h.time_stretch(x, SR, 0.85) is x


def test_silence_stays_silent_and_gets_longer():
    z = np.zeros(SR, dtype=np.float32)
    y = h.time_stretch(z, SR, 0.85)
    assert abs(len(y) / len(z) - 1 / 0.85) < 0.01 and not y.any()


def test_non_finite_samples_do_not_crash_or_leak():
    x = speechlike(1.0)
    x[100], x[5000] = np.nan, np.inf
    y = h.time_stretch(x, SR, 0.85)
    assert np.isfinite(y).all()


def test_speech_with_a_pause_keeps_the_pause():
    x = speechlike(4.0)
    y = h.time_stretch(x, SR, 0.85)
    win = int(0.04 * SR)
    quiet = [np.sqrt(np.mean(y[s:s + win] ** 2)) for s in range(0, len(y) - win, win)]
    assert min(quiet) < 0.02


def test_no_fade_in_at_the_start_and_no_clicks():
    x = speechlike(2.0)
    y = h.time_stretch(x, SR, 0.85)
    assert np.max(np.abs(np.diff(y))) < 3 * np.max(np.abs(np.diff(x)))


def test_stretch_failure_falls_back_to_the_original_audio(worker, monkeypatch):
    monkeypatch.setattr(h, "time_stretch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    en, _ = decode(run("en"))
    assert len(en) == len(worker)


def test_processing_overhead_is_small():
    x = speechlike(6.0)
    t = time.perf_counter()
    h.time_stretch(x, SR, 0.85)
    assert time.perf_counter() - t < 3.0                  # measured far below this; the bound only catches an accidental quadratic path


# ---------------------------------------------------------------------------------------------------------- 6. language normalisation, default
@pytest.mark.parametrize("tag", ["en", "EN", "english", "English", "en-US", "en_GB", "en-IN", " en "])
def test_every_english_spelling_maps_to_english(tag):
    assert h.pace_language(tag) == "en"
    assert h.pace_speed(tag, {"en": 0.85, "hi": 1.0, "other": 1.0}) == 0.85


@pytest.mark.parametrize("tag", [None, "", "xx", "fr", "ja", "ennui", "enx"])
def test_missing_or_other_languages_use_the_safe_default(tag):
    assert h.pace_speed(tag, {"en": 0.85, "hi": 1.0, "other": 1.0}) == 1.0


def test_other_languages_are_not_stretched_by_default(worker):
    assert len(decode(run("fr"))[0]) == len(worker)


# ---------------------------------------------------------------------------------------------------------- 7. configuration
def test_defaults_are_english_085_hindi_100_other_100():
    assert h.load_pace_config({}) == {"en": 0.85, "hi": 1.0, "other": 1.0}


def test_env_overrides_each_language():
    c = h.load_pace_config({"TTS_SPEED_EN": "0.9", "TTS_SPEED_HI": "0.95", "TTS_SPEED_OTHER": "1.1"})
    assert c == {"en": 0.9, "hi": 0.95, "other": 1.1}


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "nan", "inf", "0.2", "3", "0,85"])
def test_invalid_speed_configuration_fails_safe_to_no_processing(bad):
    warnings = []
    c = h.load_pace_config({"TTS_SPEED_EN": bad}, warn=warnings.append)
    assert c["en"] == 1.0 and warnings and "TTS_SPEED_EN" in warnings[0]


def test_empty_setting_means_default_not_invalid():
    warnings = []
    assert h.load_pace_config({"TTS_SPEED_EN": "  "}, warn=warnings.append)["en"] == 0.85 and not warnings


def test_speed_one_disables_english_processing(worker, monkeypatch):
    monkeypatch.setattr(h, "PACE", {"en": 1.0, "hi": 1.0, "other": 1.0})
    assert len(decode(run("en"))[0]) == len(worker)


# ---------------------------------------------------------------------------------------------------------- 8. worker contract
def test_response_schema_is_unchanged(worker):
    r = run("en")
    assert set(r) == BASELINE_KEYS and set(r["timing"]) == {"generate_s", "total_s", "audio_s"}
    assert r["format"] == "wav" and r["language"] == "en" and r["voice"] == "utkarsh"


def test_reported_audio_seconds_describe_the_audio_returned(worker):
    r = run("en")
    data, sr = decode(r)
    assert abs(r["timing"]["audio_s"] - len(data) / sr) < 0.002


def test_unsupported_language_error_is_unchanged(worker):
    assert h.handler({"input": {"text": "x", "language": "english"}}) == {"error": "Unsupported language: english"}


def test_streaming_is_still_refused(worker):
    assert h.handler({"input": {"text": "x", "language": "en", "stream": True}})["error"] == "streaming_disabled"


def test_profile_detail_records_the_original_and_the_processed_audio(worker, monkeypatch):
    monkeypatch.setattr(h._P, "enabled", True)
    d = run("en")["timing_detail"]
    assert d["speed"] == 0.85 and abs(d["raw_audio_s"] - len(worker) / SR) < 0.002 and d["audio_s"] > d["raw_audio_s"]
    assert d["raw_peak"] > 0 and d["raw_rms"] > 0 and d["stretch_s"] >= 0
