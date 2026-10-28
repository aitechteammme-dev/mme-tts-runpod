"""Runs pytest inside the built image for the build-only workflow, with the test tooling (pytest, pluggy, iniconfig) living in a SEPARATE folder (/pt, mounted read-only).

    python ci/run_tests.py [pytest arguments]      run the tests
    python ci/run_tests.py --check-tooling          prove the tooling folder is exactly what ci/requirements-test.txt pins

Import precedence, by design:
  * The tooling packages (pytest, _pytest, pluggy, iniconfig, py) are resolved from the tooling folder FIRST, by a finder placed at the front of sys.meta_path. The base image can already contain
    other versions of them (the conda base ships pluggy 1.5.0, which pytest 9.2.0 cannot use), and a path ENTRY cannot win against site-packages once it is appended; it would lose to the base copy.
  * EVERYTHING ELSE is untouched: the tooling folder is not on sys.path, so no other module can be imported from it, and the image's locked runtime packages (packaging, pygments, torch, ...) are
    resolved exactly as before. After the run, anything outside the five tooling names that came from the folder fails the run, and so does any tooling name that did not.
Package versions are read from the package METADATA found in the tooling folder (never from `module.__version__`, which many packages do not define) and compared with the pins.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import os
import re
import sys
from importlib import metadata

TOOLING_DIR = os.environ.get("MME_TEST_TOOLING_DIR", "/pt")
TOOLING_MODULES = {"pytest", "_pytest", "pluggy", "iniconfig", "py"}      # the only top-level names ever imported from the tooling folder
ALLOWED_FROM_TOOLING = TOOLING_MODULES


class ToolingFirstFinder:
    """Resolves the tooling packages from the tooling folder before any other finder; every other name, and every submodule (found through its parent's __path__), is left to the normal machinery."""

    def __init__(self, tooling_dir: str, names=TOOLING_MODULES):
        self.tooling_dir = tooling_dir
        self.names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        if path is None and fullname in self.names:
            return importlib.machinery.PathFinder.find_spec(fullname, [self.tooling_dir])
        return None


def install_tooling_first(tooling_dir: str | None = None) -> ToolingFirstFinder:
    finder = ToolingFirstFinder(tooling_dir or TOOLING_DIR)
    sys.meta_path.insert(0, finder)
    return finder


def _under(path: str | None, folder: str) -> bool:
    return bool(path) and path.startswith(os.path.join(folder, ""))


def shadowing(modules: dict, tooling_dir: str | None = None) -> list[str]:
    """Top-level modules imported from the tooling folder although they are not tooling packages."""
    folder = tooling_dir or TOOLING_DIR
    found = set()
    for name, mod in list(modules.items()):
        top = name.split(".")[0]
        if _under(getattr(mod, "__file__", None), folder) and top not in ALLOWED_FROM_TOOLING:
            found.add(top)
    return sorted(found)


def wrong_origin(modules: dict, tooling_dir: str | None = None) -> list[str]:
    """Tooling packages that are loaded but NOT from the tooling folder (e.g. the base image's pluggy)."""
    folder = tooling_dir or TOOLING_DIR
    bad = set()
    for name, mod in list(modules.items()):
        top = name.split(".")[0]
        path = getattr(mod, "__file__", None)
        if top in TOOLING_MODULES and path and not _under(path, folder):
            bad.add(f"{top} ({path})")
    return sorted(bad)


def tooling_version(dist: str, tooling_dir: str | None = None) -> str | None:
    """Version of `dist` according to the METADATA inside the tooling folder only (the base image's copy, if any, is not consulted)."""
    found = list(metadata.distributions(name=dist, path=[tooling_dir or TOOLING_DIR]))
    return found[0].version if found else None


def pinned_tooling(requirements: str | None = None) -> dict[str, str]:
    """{distribution: version} pinned in ci/requirements-test.txt (the file the tooling step installs from)."""
    path = requirements or os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements-test.txt")
    pins = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"([A-Za-z0-9_.\-]+)==([^\s\\]+)", line)
            if m:
                pins[m.group(1).lower()] = m.group(2)
    return pins


def check_tooling(requirements: str | None = None) -> int:
    """Used by the workflow's tooling step, with the network disabled. Exit 0 only if, for every pinned package, the folder's METADATA has exactly the pinned version AND the module that
    actually imports (with the tooling-first finder installed, as in the test run) lives in the folder."""
    if not os.path.isdir(TOOLING_DIR):
        print(f"ERROR: test tooling folder {TOOLING_DIR} is not mounted", file=sys.stderr)
        return 2
    preloaded = wrong_origin(sys.modules)
    if preloaded:
        print(f"ERROR: tooling packages were already imported from elsewhere before the check: {preloaded}", file=sys.stderr)
        return 3
    install_tooling_first()
    problems, report = [], []
    for dist, want in pinned_tooling(requirements).items():
        got = tooling_version(dist)
        if got is None:
            problems.append(f"{dist}: no metadata in {TOOLING_DIR}")
            continue
        if got != want:
            problems.append(f"{dist}: installed {got}, pinned {want}")
        try:
            mod = importlib.import_module(dist)
        except ImportError as exc:
            problems.append(f"{dist}: cannot be imported ({exc})")
            continue
        origin = getattr(mod, "__file__", "") or ""
        report.append(f"{dist}=={got} <- {origin}")
        if not _under(origin, TOOLING_DIR):
            problems.append(f"{dist}: imported from {origin or 'an unknown location'}, not from {TOOLING_DIR}")
    problems += [f"{x} was loaded outside the tooling folder" for x in wrong_origin(sys.modules)]
    if problems:
        print("ERROR: test tooling check failed: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"test tooling OK in {TOOLING_DIR} | interpreter {sys.executable}\n  " + "\n  ".join(report))
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--check-tooling"]:
        return check_tooling()
    if not os.path.isdir(TOOLING_DIR):
        print(f"ERROR: test tooling folder {TOOLING_DIR} is not mounted (the tooling step installs pytest there)", file=sys.stderr)
        return 2
    preloaded = wrong_origin(sys.modules)
    if preloaded:
        print(f"ERROR: tooling packages were already imported from elsewhere: {preloaded}", file=sys.stderr)
        return 3
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())   # what `python -m pytest` does: the repo checkout (cwd) is importable, as before
    install_tooling_first()               # tooling names come from the tooling folder; the folder is NOT put on sys.path
    try:
        import pytest
    except ImportError as exc:
        print(f"ERROR: pytest is not importable from {TOOLING_DIR} with {sys.executable}: {exc}", file=sys.stderr)
        return 2
    import pluggy
    print(f"pytest {tooling_version('pytest')} ({os.path.dirname(pytest.__file__)}) | pluggy {tooling_version('pluggy')} ({os.path.dirname(pluggy.__file__)}) | interpreter {sys.executable}", flush=True)
    bad = shadowing(sys.modules) + wrong_origin(sys.modules)
    if bad:
        print(f"ERROR: tooling/runtime import mix-up before the run: {bad}", file=sys.stderr)
        return 3
    rc = pytest.main(argv)
    bad = shadowing(sys.modules) + wrong_origin(sys.modules)
    if bad:
        print(f"ERROR: after the run, tooling/runtime packages were imported from the wrong place: {bad}", file=sys.stderr)
        return 3
    return int(rc)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
