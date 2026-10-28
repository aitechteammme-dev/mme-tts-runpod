"""ci/run_tests.py: the shadowing check and its failure modes (no Docker)."""
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ci_run_tests", ROOT / "ci" / "run_tests.py")
rt = importlib.util.module_from_spec(spec)
sys.modules["ci_run_tests"] = rt
spec.loader.exec_module(rt)


def mod(path):
    return types.SimpleNamespace(__file__=path)


def test_tooling_packages_from_the_tooling_folder_are_allowed_and_nothing_else():
    modules = {"pytest": mod("/pt/pytest/__init__.py"), "_pytest.config": mod("/pt/_pytest/config/__init__.py"), "pluggy": mod("/pt/pluggy/__init__.py"), "iniconfig": mod("/pt/iniconfig/__init__.py"),
               "packaging": mod("/opt/conda/lib/python3.11/site-packages/packaging/__init__.py"), "builtin": types.SimpleNamespace()}
    assert rt.shadowing(modules, "/pt") == []
    modules["packaging.version"] = mod("/pt/packaging/version.py")
    modules["pygments"] = mod("/pt/pygments/__init__.py")
    assert rt.shadowing(modules, "/pt") == ["packaging", "pygments"]


def test_a_prefix_lookalike_folder_is_not_the_tooling_folder():
    assert rt.shadowing({"packaging": mod("/ptx/packaging/__init__.py")}, "/pt") == []


def test_missing_tooling_folder_is_a_clear_failure(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "ci" / "run_tests.py")], env={"MME_TEST_TOOLING_DIR": str(tmp_path / "nope"), "PATH": ""}, capture_output=True, text=True)
    assert r.returncode == 2 and "not mounted" in r.stderr


def test_folder_without_pytest_is_a_clear_failure_and_not_a_pass(tmp_path):
    empty = tmp_path / "pt"
    empty.mkdir()
    code = ("import sys; sys.modules['pytest'] = None; sys.argv = ['run_tests.py']; "
            "import importlib.util as u; s = u.spec_from_file_location('r', %r); m = u.module_from_spec(s); s.loader.exec_module(m); sys.exit(m.main([]))" % str(ROOT / "ci" / "run_tests.py"))
    r = subprocess.run([sys.executable, "-c", code], env={"MME_TEST_TOOLING_DIR": str(empty), "PATH": ""}, capture_output=True, text=True)
    assert r.returncode == 2 and "pytest is not importable" in r.stderr


def test_tooling_dir_is_appended_not_prepended(tmp_path, monkeypatch):
    seen = {}

    class FakePytest:
        __version__ = "0"
        __file__ = str(tmp_path / "pytest" / "__init__.py")

        @staticmethod
        def main(argv):
            seen["path"] = list(sys.path)
            return 0
    monkeypatch.setattr(rt, "TOOLING_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "pytest", FakePytest)
    assert rt.main(["x"]) == 0
    assert seen["path"][-1] == str(tmp_path) and str(tmp_path) not in seen["path"][:-1]
    sys.path.remove(str(tmp_path))


def fake_tooling(tmp_path, versions=None, skip=()):
    """A tooling folder like the one `pip install --target` makes, BUT whose packages define no __version__ (iniconfig 2.3.1 does not), with real-looking dist-info metadata."""
    pins = rt.pinned_tooling()
    versions = versions or pins
    for dist, ver in versions.items():
        if dist in skip:
            continue
        pkg = tmp_path / dist
        pkg.mkdir()
        (pkg / "__init__.py").write_text("# intentionally defines no version attribute\n", encoding="utf-8")
        info = tmp_path / f"{dist}-{ver}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: {ver}\n", encoding="utf-8")
    return tmp_path


def run_check(folder):
    # -S: no site-packages, like the image where pytest is not installed; the check must work with the tooling folder alone
    return subprocess.run([sys.executable, "-S", str(ROOT / "ci" / "run_tests.py"), "--check-tooling"], env={"MME_TEST_TOOLING_DIR": str(folder), "PATH": ""}, capture_output=True, text=True)


def test_check_tooling_passes_for_packages_that_define_no_version_attribute(tmp_path):
    """REGRESSION: the workflow's first check printed `iniconfig.__version__` and crashed with AttributeError although the install had succeeded."""
    folder = fake_tooling(tmp_path)
    assert all("__version__" not in (folder / d / "__init__.py").read_text() for d in rt.pinned_tooling())
    r = run_check(folder)
    assert r.returncode == 0, r.stderr
    pins = rt.pinned_tooling()
    assert all(f"{d}=={v}" in r.stdout for d, v in pins.items())


def test_check_tooling_rejects_a_wrong_missing_or_unmounted_tooling(tmp_path):
    pins = rt.pinned_tooling()
    (tmp_path / "w").mkdir()
    r = run_check(fake_tooling(tmp_path / "w", {**pins, "pluggy": "0.0.1"}))
    assert r.returncode == 1 and "pluggy: installed 0.0.1, pinned" in r.stderr
    (tmp_path / "m").mkdir()
    r = run_check(fake_tooling(tmp_path / "m", skip=("iniconfig",)))
    assert r.returncode == 1 and "iniconfig" in r.stderr
    assert run_check(tmp_path / "absent").returncode == 2


def test_pinned_tooling_matches_the_requirements_file():
    assert rt.pinned_tooling() == {"pytest": "9.2.0", "pluggy": "1.7.0", "iniconfig": "2.3.1"}


def test_a_failing_pytest_run_stays_failing(tmp_path, monkeypatch):
    class FakePytest:
        __version__ = "0"
        __file__ = str(tmp_path / "pytest" / "__init__.py")
        main = staticmethod(lambda argv: 1)
    monkeypatch.setattr(rt, "TOOLING_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "pytest", FakePytest)
    assert rt.main([]) == 1
    sys.path.remove(str(tmp_path))
