"""Logic of ci/image_checks.py that can be tested without the image: the `pip check` classifier and the lock parser. Pure Python."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("image_checks", ROOT / "ci" / "image_checks.py")
ic = importlib.util.module_from_spec(spec)
sys.modules["image_checks"] = ic
sys.path.insert(0, str(ROOT))
spec.loader.exec_module(ic)

LOCK = {"numpy", "pillow", "scipy"}
EXPECTED = (
    "chatterbox-tts 0.1.7 requires gradio, which is not installed.\n"
    'chatterbox-tts 0.1.7 has requirement torch==2.6.0; python_version < "3.14", but you have torch 2.7.1+cu128.\n'
    'chatterbox-tts 0.1.7 has requirement torchaudio==2.6.0; python_version < "3.14", but you have torchaudio 2.7.1+cu128.\n'
)


def test_the_three_known_chatterbox_conflicts_are_allowed():
    assert ic.pip_check_findings(EXPECTED, LOCK) == ([], [])
    assert ic.pip_check_findings("No broken requirements found.\n", LOCK) == ([], [])


def test_any_other_conflict_about_our_packages_is_unexpected():
    bad = "chatterbox-tts 0.1.7 has requirement numpy<1.26, but you have numpy 1.26.4.\n"
    assert ic.pip_check_findings(EXPECTED + bad, LOCK)[0] == [bad.strip()]
    assert ic.pip_check_findings("scipy 1.17.1 requires numpy>=2, but you have numpy 1.26.4.\n", LOCK)[0]
    assert ic.pip_check_findings("torchvision 0.22.1 requires pillow<9, but you have pillow 13.0.0.\n", LOCK)[0]      # a base-image package disagreeing with a LOCKED package is our problem
    assert ic.pip_check_findings("chatterbox-tts 0.1.8 requires gradio, which is not installed.\n", LOCK)[0]            # a different Chatterbox version is not the allowed line


def test_base_image_only_conflicts_are_reported_not_failed():
    unexpected, other = ic.pip_check_findings("conda-build 3.0 requires some-base-tool, which is not installed.\n", LOCK)
    assert unexpected == [] and len(other) == 1


def test_lock_parser_handles_hash_continuations_and_normalises_names(tmp_path):
    p = tmp_path / "lock"
    p.write_text("# comment\nSome_Pkg==1.2.3 \\\n    --hash=sha256:" + "a" * 64 + " \\\n    --hash=sha256:" + "b" * 64 + "\nother.pkg==4.5\n", encoding="utf-8")
    assert ic.lock_pins(p) == {"some-pkg": "1.2.3", "other-pkg": "4.5"}


def test_the_real_lock_parses_to_159_pins_without_torch():
    pins = ic.lock_pins(ROOT / "requirements.lock")
    assert len(pins) == 159 and "torch" not in pins and "torchaudio" not in pins and pins["deprecated"] == "1.3.1"
