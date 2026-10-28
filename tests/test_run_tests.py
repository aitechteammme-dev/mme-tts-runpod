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


def test_a_failing_pytest_run_stays_failing(tmp_path, monkeypatch):
    class FakePytest:
        __version__ = "0"
        __file__ = str(tmp_path / "pytest" / "__init__.py")
        main = staticmethod(lambda argv: 1)
    monkeypatch.setattr(rt, "TOOLING_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "pytest", FakePytest)
    assert rt.main([]) == 1
    sys.path.remove(str(tmp_path))
