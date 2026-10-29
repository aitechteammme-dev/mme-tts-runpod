"""Docker build step: download ONLY the pinned files from the exact checkpoint revision, then verify them. Any mismatch exits non-zero and fails the build.

    python -m pinned.fetch_checkpoint /app/model
"""
from __future__ import annotations

import sys

from . import pins
from .assets import build_cangjie_cache
from .errors import PinnedRuntimeError
from .runtime import verify_checkpoint


def main(argv) -> int:
    if len(argv) != 2:
        print("usage: python -m pinned.fetch_checkpoint <dir>", file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download
    dest = argv[1]
    snapshot_download(repo_id=pins.CHECKPOINT_REPO, repo_type="model", revision=pins.CHECKPOINT_REVISION, allow_patterns=list(pins.CHECKPOINT_FILES), local_dir=dest)
    try:
        verify_checkpoint(dest)
        snap = build_cangjie_cache(dest)     # so the tokenizer's revision-less hf_hub_download finds the pinned Cangjie file offline
    except PinnedRuntimeError as exc:
        print(f"CHECKPOINT VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"checkpoint {pins.CHECKPOINT_REVISION} verified in {dest}: {len(pins.CHECKPOINT_FILES)} files; Cangjie cache {snap}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
