"""Runs pytest inside the built image for the build-only workflow, with the test tooling (pytest, pluggy, iniconfig) living in a SEPARATE folder (/pt, mounted read-only).

    python ci/run_tests.py [pytest arguments]

The folder is APPENDED to sys.path, never prepended (PYTHONPATH would prepend), so every package the image already has, including the locked ones, always wins and the test tooling
can never shadow or replace a locked runtime package. After the run it verifies that nothing outside the three tooling packages was imported from the folder; if something was, the run fails
even when every test passed. It uses the same `python` the tooling step installed with: it fails with a clear message if pytest is not importable from the folder.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from importlib import metadata

TOOLING_DIR = os.environ.get("MME_TEST_TOOLING_DIR", "/pt")
ALLOWED_FROM_TOOLING = {"pytest", "_pytest", "pluggy", "iniconfig", "py"}


def shadowing(modules: dict, tooling_dir: str | None = None) -> list[str]:
    """Top-level modules that were imported from the tooling folder although they are not tooling packages."""
    prefix = os.path.join(tooling_dir or TOOLING_DIR, "")
    found = set()
    for name, mod in list(modules.items()):
        path = getattr(mod, "__file__", None)
        top = name.split(".")[0]
        if path and path.startswith(prefix) and top not in ALLOWED_FROM_TOOLING:
            found.add(top)
    return sorted(found)


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
    """Proves the tooling folder holds exactly the pinned versions, read from the installed package METADATA (not from `module.__version__`: many packages, e.g. iniconfig 2.3.1,
    do not define it), and that each package really imports from that folder. Used by the workflow's tooling step, with the network disabled."""
    if not os.path.isdir(TOOLING_DIR):
        print(f"ERROR: test tooling folder {TOOLING_DIR} is not mounted", file=sys.stderr)
        return 2
    sys.path.append(TOOLING_DIR)
    prefix = os.path.join(TOOLING_DIR, "")
    problems = []
    report = []
    for dist, want in pinned_tooling(requirements).items():
        try:
            got = metadata.version(dist)
        except metadata.PackageNotFoundError:
            problems.append(f"{dist}: not installed in {TOOLING_DIR}")
            continue
        try:
            mod = importlib.import_module(dist)
        except ImportError as exc:
            problems.append(f"{dist}: cannot be imported ({exc})")
            continue
        origin = getattr(mod, "__file__", "") or ""
        report.append(f"{dist}=={got}")
        if got != want:
            problems.append(f"{dist}: installed {got}, pinned {want}")
        if not origin.startswith(prefix):
            problems.append(f"{dist}: imported from {origin or 'an unknown location'}, not from {TOOLING_DIR}")
    if problems:
        print("ERROR: test tooling check failed: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"test tooling OK in {TOOLING_DIR}: {', '.join(report)} | interpreter {sys.executable}")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--check-tooling"]:
        return check_tooling()
    if not os.path.isdir(TOOLING_DIR):
        print(f"ERROR: test tooling folder {TOOLING_DIR} is not mounted (the tooling step installs pytest there)", file=sys.stderr)
        return 2
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())   # what `python -m pytest` does: the repo checkout (cwd) is importable, as before
    sys.path.append(TOOLING_DIR)          # APPENDED: site-packages (the locked runtime) takes precedence
    try:
        import pytest
    except ImportError as exc:
        print(f"ERROR: pytest is not importable from {TOOLING_DIR} with {sys.executable}: {exc}", file=sys.stderr)
        return 2
    print(f"pytest {metadata.version('pytest')} from {os.path.dirname(pytest.__file__)} | interpreter {sys.executable}", flush=True)
    bad = shadowing(sys.modules)
    if bad:
        print(f"ERROR: tooling folder shadows runtime packages: {bad}", file=sys.stderr)
        return 3
    rc = pytest.main(argv)
    bad = shadowing(sys.modules)
    if bad:
        print(f"ERROR: after the run, runtime packages were imported from the tooling folder: {bad}", file=sys.stderr)
        return 3
    return int(rc)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
