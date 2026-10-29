"""The .dockerignore must keep every file the Dockerfile COPYs and must exclude what must never reach the daemon. Uses a small matcher for the patterns this file uses
(Docker semantics: `*` within a path segment, `**` across segments, a leading `!` re-includes, the LAST matching line wins, a match on a parent directory excludes its content)."""
import re
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def to_regex(pattern: str) -> re.Pattern:
    out, i = "", 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif c == "*":
            out += "[^/]*"
            i += 1
        elif c == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(c)
            i += 1
    return re.compile("^" + out + "$")


def rules():
    result = []
    for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        result.append((negate, to_regex(line.lstrip("!").strip("/"))))
    return result


def ignored(path: str) -> bool:
    """True if `path` (relative, posix) is excluded: evaluated for the path and each parent directory, last matching rule wins."""
    parts = path.strip("/").split("/")
    state = False
    for depth in range(1, len(parts) + 1):
        candidate = "/".join(parts[:depth])
        for negate, rx in rules():
            if rx.match(candidate):
                state = not negate
        if state and depth < len(parts):
            return True            # a parent directory is excluded: nothing below it is sent (an exception cannot re-include inside it)
    return state


def copy_sources():
    sources = []
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY "):
            args = [a for a in shlex.split(line)[1:] if not a.startswith("--")]
            sources += args[:-1]
    return sources


def test_every_copy_source_exists_and_is_not_ignored():
    sources = copy_sources()
    assert {"requirements.txt", "requirements.lock", "ci/conda_diagnostics.py", "pinned/", "ref_voice.wav", "handler.py"} <= set(sources)
    for src in sources:
        p = ROOT / src
        assert p.exists(), src
        files = [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts] if p.is_dir() else [p]
        assert files, src
        for f in files:
            assert not ignored(f.relative_to(ROOT).as_posix()), f"{f.relative_to(ROOT)} is needed by the Dockerfile but ignored"


def test_the_dockerfile_itself_is_not_ignored():
    assert not ignored("Dockerfile")


def test_unneeded_and_sensitive_paths_are_excluded():
    for p in (".git/config", ".git", ".github/workflows/build-only-v4.yml", "tests/test_pinned_runtime.py", "docs/V4_PINNING.md", "ci/image_checks.py", ".env", ".env.local", "server.pem",
              "deploy.key", "credentials.json", "secrets.yaml", "pinned/__pycache__/x.pyc", "a/b/c.pyc", ".venv/bin/python", ".pytest_cache/x", "notes.log", "README.md"):
        assert ignored(p), p
    assert not ignored("ci/conda_diagnostics.py")
