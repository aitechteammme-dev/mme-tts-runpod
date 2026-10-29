"""Docker build step: download the pinned spacy_pkuseg model archive into an EMPTY folder, verify it before extracting, and verify the extracted files. A mismatch exits non-zero and fails the build.

    python -m pinned.fetch_pkuseg /app/pkuseg        (the image sets PKUSEG_HOME to that folder)
"""
from __future__ import annotations

import sys

from . import assets, pins
from .errors import PinnedRuntimeError


def main(argv) -> int:
    if len(argv) != 2:
        print("usage: python -m pinned.fetch_pkuseg <empty dir>", file=sys.stderr)
        return 2
    try:
        assets.fetch_pkuseg(argv[1])
    except PinnedRuntimeError as exc:
        print(f"PKUSEG MODEL VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"pkuseg model {pins.PKUSEG_MODEL} ({pins.PKUSEG_ARCHIVE[1][:12]}...) verified in {argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
