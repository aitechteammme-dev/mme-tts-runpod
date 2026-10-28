"""Local, verified copies of the two assets the pinned Chatterbox tokenizer would otherwise download at run time.

  Cangjie5_TC.json   tokenizer.py: hf_hub_download(repo_id="ResembleAI/chatterbox", filename="Cangjie5_TC.json", cache_dir=<checkpoint dir>), NO revision. build_cangjie_cache() lays out a Hugging Face cache
                     inside the checkpoint dir (refs/main -> the pinned revision, snapshots/<revision>/Cangjie5_TC.json) so that call finds the pinned bytes offline.
  spacy_ontonotes    spacy_pkuseg.pkuseg() downloads <PKUSEG_HOME>/spacy_ontonotes.zip when that ZIP is absent (it never looks at the extracted folder). fetch_pkuseg() puts the verified zip and the
                     verified extracted files where the package looks, so no request is made.

Everything raises PinnedRuntimeError on the first problem. Nothing here touches the network except fetch_pkuseg(), which is an image-build step.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from . import pins
from .errors import PinnedRuntimeError


def _fail(msg: str):
    raise PinnedRuntimeError(msg)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _check_file(path: Path, size: int, digest: str, what: str):
    if not path.is_file():
        _fail(f"{what} is missing: {path}")
    if path.stat().st_size != size:
        _fail(f"{what}: size {path.stat().st_size} != pinned {size} ({path})")
    got = sha256_file(path)
    if got != digest:
        _fail(f"{what}: sha256 {got} != pinned {digest} ({path})")


# ---------------------------------------------------------------------------------------------------------- Cangjie (Hugging Face cache layout)
def cangjie_snapshot_file(model_dir, revision: str = pins.CHECKPOINT_REVISION, repo: str = pins.CHECKPOINT_REPO) -> Path:
    return Path(model_dir) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision / pins.CANGJIE_FILE


def _cangjie_pin():
    return pins.CHECKPOINT_FILES[pins.CANGJIE_FILE]


def build_cangjie_cache(model_dir, revision: str = pins.CHECKPOINT_REVISION, repo: str = pins.CHECKPOINT_REPO, pin=None) -> Path:
    """Image-build step: copy the verified checkpoint file into the cache layout hf_hub_download reads. Returns the snapshot file."""
    model_dir = Path(model_dir)
    size, digest = pin or _cangjie_pin()
    _check_file(model_dir / pins.CANGJIE_FILE, size, digest, pins.CANGJIE_FILE)
    dst = cangjie_snapshot_file(model_dir, revision, repo)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_dir / pins.CANGJIE_FILE, dst)
    refs = dst.parents[2] / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision)
    verify_cangjie_cache(model_dir, revision, repo, pin)
    return dst


def verify_cangjie_cache(model_dir, revision: str = pins.CHECKPOINT_REVISION, repo: str = pins.CHECKPOINT_REPO, pin=None) -> None:
    """The file hf_hub_download will resolve for ('main', Cangjie5_TC.json) exists and has the pinned bytes."""
    size, digest = pin or _cangjie_pin()
    f = cangjie_snapshot_file(model_dir, revision, repo)
    ref = f.parents[2] / "refs" / "main"
    if not ref.is_file() or ref.read_text().strip() != revision:
        _fail(f"Cangjie cache: {ref} must exist and name the pinned revision {revision} (otherwise the tokenizer would look for, or fetch, 'main')")
    _check_file(f, size, digest, "Cangjie cache file")


# ---------------------------------------------------------------------------------------------------------- spacy_pkuseg model
def pkuseg_paths(home):
    home = Path(home)
    return home / f"{pins.PKUSEG_MODEL}.zip", home / pins.PKUSEG_MODEL


def verify_pkuseg(home, archive=None, files=None) -> None:
    """The zip (its presence is what stops the package from downloading) and every extracted file match the pins, and the extracted folder holds nothing else."""
    archive = archive or pins.PKUSEG_ARCHIVE
    files = pins.PKUSEG_FILES if files is None else files
    zpath, mdir = pkuseg_paths(home)
    _check_file(zpath, archive[0], archive[1], "pkuseg model archive")
    if not mdir.is_dir():
        _fail(f"pkuseg model folder is missing: {mdir}")
    extra = sorted(p.name for p in mdir.iterdir() if p.name not in files)
    if extra:
        _fail(f"unexpected file(s) in {mdir}: {extra}")
    for name, (size, digest) in files.items():
        _check_file(mdir / name, size, digest, f"pkuseg model file {name}")


def fetch_pkuseg(home, url: str = pins.PKUSEG_URL, archive=None, files=None, opener=urllib.request.urlopen, timeout: int = 60) -> None:
    """Image-build step: download the archive, verify size and sha256 BEFORE anything is extracted, extract only the pinned entry names, verify them, and only then place the zip."""
    archive = archive or pins.PKUSEG_ARCHIVE
    files = pins.PKUSEG_FILES if files is None else files
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    zpath, mdir = pkuseg_paths(home)
    if zpath.exists() or mdir.exists():
        _fail(f"{home} already holds pkuseg files; refusing to reuse unverified state (use an empty directory)")
    size, digest = archive
    h, got = hashlib.sha256(), 0
    with tempfile.NamedTemporaryFile(dir=home, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with opener(url, timeout=timeout) as resp:
                for block in iter(lambda: resp.read(1 << 20), b""):
                    got += len(block)
                    if got > size:
                        _fail(f"pkuseg archive is larger than the pinned {size} bytes")
                    h.update(block)
                    tmp.write(block)
            tmp.flush()
            if got != size or h.hexdigest() != digest:
                _fail(f"pkuseg archive from {url}: size {got} sha256 {h.hexdigest()} != pinned size {size} sha256 {digest}")
            with zipfile.ZipFile(tmp_path) as z:
                names = [i.filename for i in z.infolist()]
                if sorted(names) != sorted(files):
                    _fail(f"pkuseg archive entries {sorted(names)} != pinned {sorted(files)}")
                mdir.mkdir()
                for name in files:                                     # only exact pinned names: no path from the archive is ever used
                    with z.open(name) as src, open(mdir / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            os.chmod(tmp_path, 0o644)                                  # NamedTemporaryFile is 0600: keep the baked zip readable by a non-root runtime user
            os.replace(tmp_path, zpath)
        finally:
            tmp_path.unlink(missing_ok=True)
    verify_pkuseg(home, archive, files)


def verify_pkuseg_package() -> None:
    """The installed spacy_pkuseg is the pinned version, will look in the home we verified, and would download the very archive we pinned if it were missing."""
    from importlib import metadata
    try:
        ver = metadata.version("spacy-pkuseg")
    except metadata.PackageNotFoundError:
        _fail("spacy_pkuseg is not installed: the tokenizer would silently lose Chinese segmentation")
    if ver != pins.PKUSEG_DIST_VERSION:
        _fail(f"spacy_pkuseg {ver} != pinned {pins.PKUSEG_DIST_VERSION}")
    import importlib
    cfg = importlib.import_module("spacy_pkuseg.config").config
    if cfg.model_urls.get(pins.PKUSEG_MODEL) != pins.PKUSEG_URL or cfg.model_hash.get(pins.PKUSEG_MODEL) != pins.PKUSEG_ARCHIVE[1]:
        _fail("spacy_pkuseg's own URL/hash for its default model differ from our pins")
