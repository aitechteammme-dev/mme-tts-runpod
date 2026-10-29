# v4 pinning (CANDIDATE, not GPU-validated)

Branch `tts-v4-streaming`, from `main` bf6c09caf4fe5799b456901b369492320a6c2bfa. Nothing here has been built, pushed or run on a GPU.

## Pins (single source: `pinned/pins.py`)
| Item | Pin |
|---|---|
| Chatterbox source | `5de7a54aa4e5e2baadb0182dde554908b48b85c2` (chatterbox-tts 0.1.7), installed `--no-deps` |
| Perth | `ff1c8ac55a976971245cdd53c18d6131ca00d993` (upstream declares `@master`, a moving ref) |
| Checkpoint | `ResembleAI/chatterbox` @ `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` |
| T3 | `t3_mtl23ls_v3.safetensors` (explicit filename; library default is V2) |
| S3Gen | `s3gen.pt` (the pinned source loads this, NOT `s3gen_v3.*`) |
| torch / torchaudio | 2.7.1 (approved base image `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`) |
| transformers | 5.2.0 |

Checkpoint files verified by size and SHA-256 (the four LFS hashes equal those published by the Hub; the two JSONs' git blob ids equal the Hub API's):
t3 v3 `5abca832…5953`, s3gen.pt `9b9ff07e…d2a3`, ve.pt `4b16d836…79f1`, conds.pt `6552d705…7f4e`, grapheme json `69632f47…7dbf`, Cangjie json `7073fd9d…ce8c`. Full values in `pins.py`.

## Fail-closed behaviour (`pinned/runtime.py`)
Start-up refuses (raises, worker does not start) when: `T3_MODEL` is anything but the V3 file; chatterbox/Perth are not installed from the pinned commits; torch/torchaudio/transformers differ; any of 11 upstream functions
differs from its fingerprint; any checkpoint file is missing/wrong size/wrong hash; a second T3 file is present; `from_local` lacks `t3_model`; the model has no real `PerthImplicitWatermarker` or overrides `generate()`.
`input.stream` is refused with `streaming_disabled` and no audio. `STREAMING_ALLOWED=False` is a source constant, not configurable. Full-waveform output keeps upstream's whole-waveform Perth watermark.

## Open issues
1. **torch**: Chatterbox declares `torch==2.6.0`, `torchaudio==2.6.0`. We keep the approved 2.7.1. Import and fingerprints pass on CPU torch 2.7.1; whether the models run identically on GPU is unverified.
2. **Python**: lock produced on Python 3.12 CPU; the image Python (expected 3.11) is unverified. `requirements.lock` is a candidate; the authoritative freeze is `/app/pip-freeze.txt` from the first GPU build.
3. **Perth** installs with `--no-deps`; its dependencies come from the lock.
4. **No chunked watermark method is verified.** CPU experiment (n=1 voice, 8 s): detector score stays 1.0 when chunks are watermarked separately, but the join discontinuity grows (+35% at 1 s chunks, +65% at 0.2 s). Evidence for a product decision only.
5. Start-up hashes ~3.2 GB each boot (measured 160 s on a loaded 5 GB WSL box; GPU host time unmeasured). May need a marker-file optimisation if cold start matters.
6. The model was not loaded on CPU (5 GB RAM); `from_local` with the V3 file is unexercised.
7. ~~`verify_distributions` untested against a real git install~~ Resolved in review: a fresh venv installed with the Dockerfile's exact `pip install --no-deps git+...@<sha>` commands records `direct_url.json` commit ids equal to the pins, and chatterbox/perth checks pass (torch was absent there).

## Pre-GPU review additions (2026-10-28)
- `requirements.lock`: `Deprecated==3.0.0` requires Python >= 3.12 and would fail the image build on 3.11 (pytorch v2.7.1 `docker.Makefile` builds with PYTHON_VERSION 3.11). Changed to `1.3.1` (needed only by pykakasi). After the change all 159 pins have cp311 linux x86_64 wheels and the Requires-Dist closure is consistent (PyPI metadata, markers evaluated for 3.11). Not installed in the actual image.
- RESOLVED in the next section (found in review): Chatterbox's tokenizer does two network downloads outside our pins: (a) `Cangjie5_TC.json` via `hf_hub_download` with no revision and `cache_dir=<checkpoint dir>`: with `HF_HUB_OFFLINE=1` it loads 0 entries (Chinese Cangjie silently degraded, only a warning); online it would fetch `main`; (b) `spacy_pkuseg` downloads `spacy_ontonotes.zip` from GitHub into `~/.pkuseg` at every cold start, unpinned and unhashed. Neither is baked or verified in this Dockerfile.

## Tokenizer assets (no run-time downloads)
Both downloads are replaced by build-time, hash-verified copies. Pins are in `pinned/pins.py`, logic in `pinned/assets.py`, checks in `pinned/runtime.py`.

| Asset | Source | Pin | How the pinned tokenizer finds it |
|---|---|---|---|
| `Cangjie5_TC.json` | `ResembleAI/chatterbox` @ `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` (same checkpoint revision; the tokenizer asks for REPO_ID `ResembleAI/chatterbox`, no revision) | 1920163 B, sha256 `7073fd9d…ce8c`; git blob `d77891f84ca1db0d6f7058a4ee081d4bb0bfe88e` equals the Hub API's | `fetch_checkpoint` lays out a Hugging Face cache inside the checkpoint dir (`models--ResembleAI--chatterbox/refs/main` -> the pinned revision, `snapshots/<rev>/Cangjie5_TC.json`); with `HF_HUB_OFFLINE=1` the revision-less `hf_hub_download(cache_dir=<checkpoint dir>)` resolves to it |
| `spacy_ontonotes` (pkuseg model) | `https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip` (release tag commit `9f14dc133897cc30a5770e8892f58aaf27cb8db1`, published 2020-10-05; the release is NOT marked immutable and the asset has no GitHub digest) | 34567143 B, sha256 `b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe` | `fetch_pkuseg` downloads into an empty `PKUSEG_HOME=/app/pkuseg`, verifies size+sha256 BEFORE extracting, extracts only `features.msgpack` (22685181 B, `fd432248…a18`) and `weights.npz` (37508754 B, `5ada075e…c9e`), re-verifies. `spacy_pkuseg` skips its download when `<PKUSEG_HOME>/spacy_ontonotes.zip` exists, so the verified zip stays in place |

Evidence for the pkuseg hash: it is the `model_hash["spacy_ontonotes"]` shipped inside the pinned `spacy_pkuseg==1.0.2` and equals `sha256sum` of two independent downloads (the package's own, and `fetch_pkuseg`'s). `verify_pkuseg_package` fails if the installed package's URL/hash/version ever differ from these pins. Trust anchor = the package author's hash; GitHub's release is mutable, so the hash is the control.

Run-time guarantees: start-up requires `HF_HUB_OFFLINE=1` (effective in `huggingface_hub.constants`) and `PKUSEG_HOME` equal to what `spacy_pkuseg` actually uses; verifies both assets; runs `from_local` inside `runtime.no_network()` (any INET connect or DNS lookup is refused, recorded and fails the start-up even if a library swallows the error); and after loading requires the Cangjie mapping to hold exactly 99228 words and the pkuseg segmenter to exist. The Docker build runs the same tokenizer initialization with the network denied.

Negative controls (CPU): without the Cangjie cache layout the tokenizer, offline, silently loads 0 Cangjie words (the original defect; now caught by the word-count check); with an empty `PKUSEG_HOME` the tokenizer tries to reach `github.com` (now refused by `no_network`).

Not covered (out of scope, report only): Chatterbox's lazy, optional `dicta_onnx` (Hebrew) and `russian_text_stresser` (Russian) imports are not installed by the lock; those languages are silently skipped with a log warning.

## Stage 0: build-only CI check (prepared, NOT yet run)
`.github/workflows/build-only-v4.yml` builds the Dockerfile and checks the image without ever publishing it.
- **Triggers:** push to branch `tts-v4-streaming` (docs-only pushes ignored) or manual `workflow_dispatch` (GitHub lists manual workflows only once the file exists on the default branch). It does not run for any other branch. `build-image.yml` (the publisher, `tts-v3` only) is unchanged.
- **Permissions:** `contents: read` only. No `packages: write`, no secrets, no registry login, no `docker push`/`tag`, no `build-push-action`. Checkout keeps no credentials. Only `actions/checkout` and `actions/upload-artifact`, both pinned by commit SHA.
- **Image:** built under the local-only tag `mme-tts-v4-buildcheck:run-<run_id>-<attempt>` (no registry host), removed at the end.
- **Artifacts (GitHub workflow artifacts, 7 days, text only):** `build.log`, `disk.txt`, `image-checks.json/.log`, `pip-freeze.txt`, `pip-check.txt`, `pytest.log`. No registry or package write is possible.
- **Disk/time:** preflight removes unused preinstalled toolchains and requires >= 35 GB free on the Docker filesystem before building (estimate ~20 GB image + cache; GitHub documents only 14 GB "SSD" for standard runners, root free space is not documented, so the check is mandatory). Timeouts: build 50 min, checks 15, tests 15, job 75.
- **What it enforces:** base image pinned by digest and asserted to be Python 3.11; `pip install --require-hashes --no-deps` of the lock (3747 sha256 entries); checkpoint / Cangjie / pkuseg hashes; pinned commits and fingerprints; the tokenizer smoke test inside `RUN --network=none`; then in containers started with `--network none`: `ci/image_checks.py` (torch 2.7.1 + CUDA 12.8 build, every lock pin installed at its locked version, `pip check` limited to the 3 known Chatterbox conflicts, checkpoint re-hash, tokenizer offline) and the full test suite against the real baked assets.
- **Hardening done:** base image digest pin `sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2` (amd64, 4.22 GB compressed, tag last pushed 2025-06-04); hash-locked `requirements.lock` (hashes of every PyPI file of each pinned version; verified on CPython 3.11.18 that the hashed lock installs, a tampered hash is rejected, and `pip check` shows only the 3 known conflicts).
- **Not verified:** the workflow has never run; `actionlint` was not run (downloading a binary was not permitted); Docker is not installed here, so `RUN --network=none` was exercised only through `unshare -rn` (a real network namespace), not through BuildKit.

### Stage 0 hardening (second pass, still not run)
- **No torch reinstall.** Evidence from the base image at the pinned digest (read from its layer's tar index, nothing pulled or run): torch 2.7.1+cu128, torchaudio 2.7.1+cu128, torchvision 0.22.1+cu128, nvidia-cuda-runtime-cu12 12.8.57, all pip-owned with RECORD; the pytorch v2.7.1 Dockerfile pip-installs them from the cu128 index. The Dockerfile now asserts exactly `torch == 2.7.1+cu128`, `torch.version.cuda == 12.8` and `torchaudio == 2.7.1+cu128` at the top and again after all installs; `ci/image_checks.py` asserts the same on the built image. A different base fails the build.
- **Conda diagnostics.** `ci/conda_diagnostics.py` runs before the hash-locked install and changes nothing. On the real base data: 36 distributions overlap the lock, 31 would be replaced (conda- and pip-owned, including numpy 2.2.6 -> 1.26.4), 5 are already at the locked version (colorama, jinja2, mpmath, six, sympy: pip will not re-check their hashes), and none lacks RECORD, so pip can uninstall all of them. The step fails the build early only if pip is certain to fail.
- **Failure evidence.** Evidence, image-check and test steps run whenever the image was built (`!cancelled() && steps.build.outcome == 'success'`), summary and artifact upload run always; there is no `continue-on-error`, so a failure still fails the job.
- **`.dockerignore`.** Excludes `.git`, `.github`, caches, virtual environments, logs, credentials/keys, docs, tests and `ci/*` except `ci/conda_diagnostics.py`. `tests/test_dockerignore.py` checks every file the Dockerfile COPYs stays in the context.
