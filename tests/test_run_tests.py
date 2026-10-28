"""ci/run_tests.py: import precedence of the test tooling, its checks and failure modes (no Docker)."""
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ci" / "run_tests.py"
spec = importlib.util.spec_from_file_location("ci_run_tests", SCRIPT)
rt = importlib.util.module_from_spec(spec)
sys.modules["ci_run_tests"] = rt
spec.loader.exec_module(rt)


def mod(path):
    return types.SimpleNamespace(__file__=path)


def make_pkg(folder: Path, dist: str, version: str, marker: str = ""):
    """A package like `pip install --target` makes: <dist>/__init__.py (defining NO __version__, like iniconfig 2.3.1) and <dist>-<version>.dist-info/METADATA."""
    pkg = folder / dist
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"# intentionally defines no version attribute\nMARKER = {marker!r}\n", encoding="utf-8")
    info = folder / f"{dist}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n", encoding="utf-8")


def fake_tooling(folder: Path, versions=None, skip=()):
    folder.mkdir(parents=True, exist_ok=True)
    for dist, ver in (versions or rt.pinned_tooling()).items():
        if dist not in skip:
            make_pkg(folder, dist, ver, marker="tooling")
    return folder


def run_script(args, tooling, pythonpath=None, code=None):
    """Fresh interpreter with -S (no site-packages, like the image where pytest is not installed). `pythonpath` stands in for the base image's site-packages, AHEAD of everything."""
    env = {"MME_TEST_TOOLING_DIR": str(tooling), "PATH": ""}
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    cmd = [sys.executable, "-S", *(["-c", code] if code else [str(SCRIPT), *args])]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


# ---------------------------------------------------------------------------------------------------------- shadowing / origin checks
def test_tooling_packages_from_the_tooling_folder_are_allowed_and_nothing_else():
    modules = {"pytest": mod("/pt/pytest/__init__.py"), "_pytest.config": mod("/pt/_pytest/config/__init__.py"), "pluggy": mod("/pt/pluggy/__init__.py"), "iniconfig": mod("/pt/iniconfig/__init__.py"),
               "packaging": mod("/opt/conda/lib/python3.11/site-packages/packaging/__init__.py"), "builtin": types.SimpleNamespace()}
    assert rt.shadowing(modules, "/pt") == [] and rt.wrong_origin(modules, "/pt") == []
    modules["packaging.version"] = mod("/pt/packaging/version.py")
    modules["pygments"] = mod("/pt/pygments/__init__.py")
    assert rt.shadowing(modules, "/pt") == ["packaging", "pygments"]


def test_a_tooling_package_loaded_from_the_base_is_reported():
    modules = {"pluggy": mod("/opt/conda/lib/python3.11/site-packages/pluggy/__init__.py"), "pytest": mod("/pt/pytest/__init__.py")}
    assert rt.wrong_origin(modules, "/pt") == ["pluggy (/opt/conda/lib/python3.11/site-packages/pluggy/__init__.py)"]


def test_a_prefix_lookalike_folder_is_not_the_tooling_folder():
    assert rt.shadowing({"packaging": mod("/ptx/packaging/__init__.py")}, "/pt") == []
    assert rt.wrong_origin({"pluggy": mod("/ptx/pluggy/__init__.py")}, "/pt") != []


# ---------------------------------------------------------------------------------------------------------- the finder (in-process, unique module names)
def test_tooling_first_finder_beats_the_base_and_leaves_every_other_module_alone(tmp_path):
    base, tooling = tmp_path / "base", tmp_path / "tooling"
    make_pkg(base, "zz_tool", "1.5.0", marker="base")                 # on sys.path, like site-packages
    make_pkg(tooling, "zz_tool", "1.7.0", marker="tooling")
    make_pkg(tooling, "zz_other", "9.9.9", marker="tooling")          # only in the tooling folder, and NOT a tooling name
    sys.path.insert(0, str(base))
    finder = rt.ToolingFirstFinder(str(tooling), names={"zz_tool"})
    sys.meta_path.insert(0, finder)
    try:
        import zz_tool
        assert zz_tool.MARKER == "tooling" and str(tooling) in zz_tool.__file__           # the base copy is ahead on sys.path, and still loses
        try:
            import zz_other  # noqa: F401
            raise AssertionError("a non-tooling module was importable from the tooling folder")
        except ImportError:
            pass
    finally:
        sys.meta_path.remove(finder)
        sys.path.remove(str(base))
        for name in ("zz_tool", "zz_other"):
            sys.modules.pop(name, None)


def test_tooling_folder_is_never_put_on_sys_path(tmp_path, monkeypatch):
    seen = {}

    class FakePytest:
        __file__ = str(tmp_path / "pytest" / "__init__.py")

        @staticmethod
        def main(argv):
            seen["path"] = list(sys.path)
            return 0
    monkeypatch.setattr(rt, "TOOLING_DIR", str(tmp_path))
    monkeypatch.setattr(rt, "install_tooling_first", lambda *a, **k: None)           # the real pluggy of this test process is not under tmp_path
    monkeypatch.setattr(rt, "wrong_origin", lambda *a, **k: [])
    monkeypatch.setitem(sys.modules, "pytest", FakePytest)
    assert rt.main(["x"]) == 0
    assert str(tmp_path) not in seen["path"]


def test_a_failing_pytest_run_stays_failing(tmp_path, monkeypatch):
    class FakePytest:
        __file__ = str(tmp_path / "pytest" / "__init__.py")
        main = staticmethod(lambda argv: 1)
    monkeypatch.setattr(rt, "TOOLING_DIR", str(tmp_path))
    monkeypatch.setattr(rt, "install_tooling_first", lambda *a, **k: None)
    monkeypatch.setattr(rt, "wrong_origin", lambda *a, **k: [])
    monkeypatch.setitem(sys.modules, "pytest", FakePytest)
    assert rt.main([]) == 1


# ---------------------------------------------------------------------------------------------------------- the exact CI failure, in a fresh interpreter
def test_REGRESSION_base_pluggy_1_5_0_must_not_override_the_pinned_pluggy_1_7_0(tmp_path):
    """CI: `pluggy==1.5.0` was imported from /opt/conda/.../site-packages instead of the pinned 1.7.0 in /pt. Here the base copy is FIRST on sys.path and has its own 1.5.0 metadata."""
    base = tmp_path / "base"
    make_pkg(base, "pluggy", "1.5.0", marker="base")
    tooling = fake_tooling(tmp_path / "pt")
    # negative control: the former approach (append the folder to sys.path) really does pick the base copy in this setup
    old = run_script([], tooling, pythonpath=base, code=f"import sys; sys.path.append({str(tooling)!r}); import pluggy; print(pluggy.MARKER)")
    assert old.stdout.strip() == "base", old.stderr
    r = run_script(["--check-tooling"], tooling, pythonpath=base)
    assert r.returncode == 0, r.stderr
    assert "pluggy==1.7.0" in r.stdout and "pluggy==1.5.0" not in r.stdout
    assert str(tooling / "pluggy" / "__init__.py") in r.stdout                        # the module that imported lives in the tooling folder


def test_check_tooling_fails_if_a_base_copy_was_already_imported(tmp_path):
    base = tmp_path / "base"
    make_pkg(base, "pluggy", "1.5.0", marker="base")
    tooling = fake_tooling(tmp_path / "pt")
    code = (f"import sys; sys.path.insert(0, {str(base)!r}); import pluggy; "
            f"import importlib.util as u; s = u.spec_from_file_location('rt', {str(SCRIPT)!r}); m = u.module_from_spec(s); s.loader.exec_module(m); sys.exit(m.check_tooling())")
    r = run_script([], tooling, pythonpath=base, code=code)
    assert r.returncode == 3 and "already imported from elsewhere" in r.stderr and "pluggy" in r.stderr


def test_check_tooling_passes_for_packages_that_define_no_version_attribute(tmp_path):
    """REGRESSION (earlier CI run): the check read `iniconfig.__version__`, which iniconfig 2.3.1 does not define."""
    tooling = fake_tooling(tmp_path)
    assert all("__version__" not in (tooling / d / "__init__.py").read_text() for d in rt.pinned_tooling())
    r = run_script(["--check-tooling"], tooling)
    assert r.returncode == 0, r.stderr
    assert all(f"{d}=={v}" in r.stdout for d, v in rt.pinned_tooling().items())


def test_check_tooling_rejects_a_wrong_missing_or_unmounted_tooling(tmp_path):
    pins = rt.pinned_tooling()
    r = run_script(["--check-tooling"], fake_tooling(tmp_path / "w", {**pins, "pluggy": "0.0.1"}))
    assert r.returncode == 1 and "pluggy: installed 0.0.1, pinned" in r.stderr
    r = run_script(["--check-tooling"], fake_tooling(tmp_path / "m", skip=("iniconfig",)))
    assert r.returncode == 1 and "iniconfig" in r.stderr
    assert run_script(["--check-tooling"], tmp_path / "absent").returncode == 2


def test_metadata_comes_from_the_tooling_folder_not_from_the_base(tmp_path):
    base = tmp_path / "base"
    make_pkg(base, "pluggy", "1.5.0")
    tooling = fake_tooling(tmp_path / "pt")
    code = (f"import sys; sys.path.insert(0, {str(base)!r}); import importlib.util as u; s = u.spec_from_file_location('rt', {str(SCRIPT)!r}); m = u.module_from_spec(s); s.loader.exec_module(m); "
            f"print(m.tooling_version('pluggy'), m.tooling_version('nonexistent'))")
    r = run_script([], tooling, pythonpath=base, code=code)
    assert r.stdout.split() == [rt.pinned_tooling()["pluggy"], "None"], r.stderr


def test_pinned_tooling_matches_the_requirements_file():
    assert rt.pinned_tooling() == {"pytest": "9.2.0", "pluggy": "1.7.0", "iniconfig": "2.3.1"}


# ---------------------------------------------------------------------------------------------------------- failure modes of the runner itself
def test_missing_tooling_folder_is_a_clear_failure(tmp_path):
    r = run_script([], tmp_path / "nope")
    assert r.returncode == 2 and "not mounted" in r.stderr


def test_folder_without_pytest_is_a_clear_failure_and_not_a_pass(tmp_path):
    empty = tmp_path / "pt"
    empty.mkdir()
    r = run_script([], empty)
    assert r.returncode == 2 and "pytest is not importable" in r.stderr
