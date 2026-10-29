"""CPU-only checks of the BUILT v4 image, run inside it by .github/workflows/build-only-v4.yml (no GPU, no network, no registry):

    docker run --rm --network none -v "$PWD":/work:ro -v "$OUT":/out IMAGE python /work/ci/image_checks.py --out /out/image-checks.json

Exit code 0 only if every check passed. Nothing here loads the 3.2 GB model; it proves the environment, the pins and the baked assets.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# What `pip check` is EXPECTED to say about the pinned Chatterbox (it declares torch/torchaudio 2.6.0 and gradio; the approved runtime is torch 2.7.1 and gradio is not needed to serve).
ALLOWED_PIP_CHECK = [
    re.compile(r"^chatterbox-tts 0\.1\.7 requires gradio, which is not installed\.$"),
    re.compile(r'^chatterbox-tts 0\.1\.7 has requirement torch==2\.6\.0; python_version < "3\.14", but you have torch 2\.7\.1(\+\S+)?\.$'),
    re.compile(r'^chatterbox-tts 0\.1\.7 has requirement torchaudio==2\.6\.0; python_version < "3\.14", but you have torchaudio 2\.7\.1(\+\S+)?\.$'),
]
results: list[dict] = []


def check(name: str, ok: bool, detail="") -> bool:
    results.append({"name": name, "ok": bool(ok), "detail": str(detail)[:600]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return bool(ok)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"([A-Za-z0-9_.\-]+)==([^\s\\]+)", line)
        if m:
            pins[norm(m.group(1))] = m.group(2)
    return pins


def pip_check_findings(output: str, lock: set[str]) -> tuple[list[str], list[str]]:
    """(unexpected lines about OUR packages, other lines). Conflicts that exist only inside the base image are reported but are not ours to fail on."""
    ours = lock | {"chatterbox-tts", "resemble-perth", "torch", "torchaudio"}
    unexpected, other = [], []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("No broken requirements"):
            continue
        if any(rx.match(line) for rx in ALLOWED_PIP_CHECK):
            continue
        first = norm(line.split()[0])
        mentioned = {norm(w.strip(",;.")) for w in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]+", line)}
        (unexpected if first in ours or (mentioned & ours and " requires " in line) else other).append(line)
    return unexpected, other


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/app/model"))
    args = ap.parse_args()
    t0 = time.time()

    check("Python is 3.11", sys.version_info[:2] == (3, 11), sys.version.split()[0])
    import torch
    import torchaudio
    from pinned import pins, runtime
    check("torch is 2.7.1 built for CUDA 12.8", torch.__version__ == "2.7.1+cu128" and torch.version.cuda == "12.8", f"{torch.__version__} cuda {torch.version.cuda}")
    check("torchaudio is 2.7.1+cu128", torchaudio.__version__ == "2.7.1+cu128", torchaudio.__version__)

    pinned = lock_pins(ROOT / "requirements.lock")
    drift = []
    for name, ver in pinned.items():
        try:
            got = metadata.version(name)
        except metadata.PackageNotFoundError:
            got = None
        if got != ver:
            drift.append(f"{name}: lock {ver}, installed {got}")
    check(f"every one of the {len(pinned)} lock pins is installed at exactly its locked version", not drift, "; ".join(drift[:8]))

    proc = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    unexpected, other = pip_check_findings(proc.stdout + proc.stderr, set(pinned))
    check("pip check reports nothing about our packages beyond the 3 known Chatterbox pin conflicts", not unexpected, "; ".join(unexpected[:6]))
    if other:
        print("note: pip check lines about the base image only (not failing):", *other[:10], sep="\n  ")

    for label, fn in (("installed packages come from the pinned commits and versions", runtime.verify_distributions),
                      ("installed Chatterbox source matches the 11 pinned fingerprints", runtime.verify_source_fingerprints)):
        try:
            fn()
            check(label, True)
        except runtime.PinnedRuntimeError as e:
            check(label, False, e)
    t = time.time()
    try:
        runtime.verify_checkpoint(args.model_dir)
        check("baked checkpoint files match the pinned sizes and sha256", True, f"{time.time() - t:.0f} s")
    except runtime.PinnedRuntimeError as e:
        check("baked checkpoint files match the pinned sizes and sha256", False, e)
    check("the V3 T3 file is the only T3 file baked", [os.path.basename(p) for p in glob.glob(f"{args.model_dir}/t3_*")] == [pins.T3_FILE])
    try:
        n = runtime.verify_tokenizer_offline(args.model_dir)
        check("tokenizer initializes offline with the Cangjie mapping and the pkuseg segmenter", n == pins.CANGJIE_DISTINCT_WORDS, f"{n} Cangjie words")
    except Exception as e:      # noqa: BLE001: any failure here IS the result
        check("tokenizer initializes offline with the Cangjie mapping and the pkuseg segmenter", False, f"{type(e).__name__}: {e}")
    check("streaming is disabled by constant and refuses requests", pins.STREAMING_ALLOWED is False and (runtime.streaming_refusal() or {}).get("error") == "streaming_disabled")

    ok = all(r["ok"] for r in results)
    summary = {"passed": ok, "seconds": round(time.time() - t0, 1), "checks": results, "pip_check_other": other[:20]}
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{sum(r['ok'] for r in results)}/{len(results)} image checks passed in {summary['seconds']} s" + ("" if ok else "  -> FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
