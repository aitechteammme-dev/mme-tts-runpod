"""Build-time diagnostic, run BEFORE the hash-locked pip install: which installed distributions overlap with requirements.lock, who owns them, and could pip uninstall them?

The base image is a Miniforge/conda environment. `pip install --require-hashes --no-deps -r requirements.lock` replaces an installed distribution of a different version by uninstalling it first,
and pip cannot uninstall one that has no RECORD file (a distutils/egg-info install). This prints one line per overlapping distribution so the build log shows the whole picture,
and exits 1 ONLY when pip is certain to fail: a distribution that must be replaced (its version differs from the lock) and has no RECORD. It never changes anything and never edits the lock.

    python ci/conda_diagnostics.py requirements.lock [--conda-meta /opt/conda/conda-meta]
"""
from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"([A-Za-z0-9_.\-]+)==([^\s\\]+)", line.lstrip("﻿"))
        if m:
            pins[norm(m.group(1))] = m.group(2)
    return pins


def conda_owned(conda_meta: Path | None) -> set[str]:
    if conda_meta is None or not conda_meta.is_dir():
        return set()
    owned = set()
    for p in conda_meta.glob("*.json"):
        m = re.match(r"^(.+)-[^-]+-[^-]+\.json$", p.name)
        if m:
            owned.add(norm(m.group(1)))
    return owned


def diagnose(pins: dict[str, str], dists, owned: set[str]) -> list[dict]:
    """One row per installed distribution whose name is in the lock."""
    rows = []
    for d in dists:
        name = norm(d.metadata["Name"] or "")
        if name not in pins:
            continue
        path = str(getattr(d, "_path", "") or "")
        has_record = d.read_text("RECORD") is not None
        installer = (d.read_text("INSTALLER") or "").strip() or "unknown"
        egg = path.endswith(".egg-info")
        replaced = d.version.split("+")[0] != pins[name]
        risk = replaced and not has_record
        rows.append({"name": name, "installed": d.version, "locked": pins[name], "replaced": replaced, "installer": installer, "conda_meta": name in owned,
                     "has_record": has_record, "egg_info": egg, "risk": risk})
    return sorted(rows, key=lambda r: r["name"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lock")
    ap.add_argument("--conda-meta", default=str(Path(sys.prefix) / "conda-meta"))
    args = ap.parse_args(argv)
    pins = lock_pins(Path(args.lock))
    rows = diagnose(pins, metadata.distributions(), conda_owned(Path(args.conda_meta)))
    print(f"python {sys.version.split()[0]} | prefix {sys.prefix} | conda-meta {'present' if Path(args.conda_meta).is_dir() else 'absent'} | lock pins {len(pins)} | overlapping installed distributions {len(rows)}")
    print(f"{'package':32s} {'installed':14s} {'locked':12s} {'action':8s} {'owner':10s} {'RECORD':7s} note")
    for r in rows:
        owner = "conda" if r["conda_meta"] else r["installer"]
        note = "pip cannot uninstall this" if r["risk"] else ("" if r["replaced"] else "same version: pip will not re-check its hash")
        print(f"{r['name']:32s} {r['installed']:14s} {r['locked']:12s} {'REPLACE' if r['replaced'] else 'keep':8s} {owner:10s} {'yes' if r['has_record'] else 'NO':7s} {note}")
    risky = [r for r in rows if r["risk"]]
    if risky:
        print(f"::error::{len(risky)} installed distribution(s) must be replaced but have no RECORD metadata, so pip cannot uninstall them: {', '.join(r['name'] for r in risky)}", file=sys.stderr)
        return 1
    print(f"ok: {sum(r['replaced'] for r in rows)} distribution(s) will be replaced, each has RECORD metadata; {sum(not r['replaced'] for r in rows)} already at the locked version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
