# The tag is pinned by digest (the linux/amd64 image Docker Hub served for this tag on 2026-10-28; the tag was last pushed 2025-06-04), so the base cannot change under a rebuild.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2

# Fail at once if the base image is not the Python the lock was checked for (3.11; pytorch v2.7.1 docker.Makefile builds this image with PYTHON_VERSION 3.11).
RUN python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; print('python', sys.version.split()[0])"

# The base image already contains the exact PyTorch stack this worker needs. Evidence (2026-10-28): the pytorch v2.7.1 Dockerfile pip-installs torch, torchvision and torchaudio from the cu128 index into /opt/conda,
# and the base layer of the pinned digest holds torch 2.7.1+cu128, torchaudio 2.7.1+cu128, torchvision 0.22.1+cu128 and nvidia-cuda-runtime-cu12 12.8.57, all pip-owned with RECORD metadata.
# So nothing is reinstalled; a different base fails the build here instead of being silently replaced. (CUDA 12.8 is the build torch was compiled for; torch.version.cuda reports it without a GPU.)
RUN python -c "import torch, torchaudio; assert torch.__version__ == '2.7.1+cu128' and torch.version.cuda == '12.8', (torch.__version__, torch.version.cuda); assert torchaudio.__version__ == '2.7.1+cu128', torchaudio.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__)"

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Pinned, in this order. Values must match pinned/pins.py (the worker re-checks them at start-up and refuses to run on a mismatch).
COPY requirements.txt requirements.lock ./
COPY ci/conda_diagnostics.py /tmp/conda_diagnostics.py

# Diagnostic (changes nothing): the base is a conda environment and the lock replaces some packages that conda or pip installed there. This prints every overlapping distribution with its owner and
# RECORD metadata, and fails the build early, with the whole list, only if pip is certain to fail (a package that must be replaced but has no RECORD, so pip cannot uninstall it).
RUN python /tmp/conda_diagnostics.py requirements.lock

# --require-hashes: every artifact must match a sha256 in the lock; --no-deps: the lock is the complete, closure-checked set (torch, torchaudio and CUDA 12.8 come from the base image, asserted above and again below).
RUN pip install --no-cache-dir --require-hashes --no-deps -r requirements.lock

# Perth (the watermarker) and Chatterbox from exact commits, --no-deps: Chatterbox declares torch==2.6.0 / torchaudio==2.6.0 and gradio, which must not replace the approved torch 2.7.1 baseline.
RUN pip install --no-cache-dir --no-deps "git+https://github.com/resemble-ai/Perth.git@ff1c8ac55a976971245cdd53c18d6131ca00d993" \
 && pip install --no-cache-dir --no-deps "git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2"

# No torch reinstall: nothing above touched torch or torchaudio (the lock has neither, and every install is --no-deps). Prove it, with the same exact assertion as at the top of this file.
RUN python -c "import torch, torchaudio; assert torch.__version__ == '2.7.1+cu128' and torch.version.cuda == '12.8', (torch.__version__, torch.version.cuda); assert torchaudio.__version__ == '2.7.1+cu128', torchaudio.__version__; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__, '(unchanged by the installs)')"

COPY pinned/ /app/pinned/

# Bake the checkpoint from the EXACT Hugging Face revision (never "main") and verify size + sha256 of every file: a mismatch fails the build.
RUN python -m pinned.fetch_checkpoint /app/model

# The tokenizer would otherwise download two more things at run time (Cangjie5_TC.json from Hub "main", and the spacy_pkuseg model zip from GitHub): bake both from pinned, hash-verified sources.
# (The Cangjie file is already in /app/model; fetch_checkpoint laid out the Hugging Face cache the tokenizer's hf_hub_download call reads.)
RUN python -m pinned.fetch_pkuseg /app/pkuseg

# From here on nothing may reach the network for models. PKUSEG_HOME must be set before spacy_pkuseg is first imported.
ENV MODEL_DIR=/app/model
ENV HF_HUB_OFFLINE=1
ENV PKUSEG_HOME=/app/pkuseg

# Build-time proof (needs no GPU): the installed packages and Chatterbox source are the pinned ones, and the tokenizer initializes with the network DENIED, with the Cangjie mapping and pkuseg segmenter loaded.
# `--network=none` (BuildKit) gives this step NO network interface, in addition to the in-process socket guard. A builder without BuildKit rejects the flag and the build fails, which is the safe outcome.
RUN --network=none python -c "from pinned import runtime; runtime.verify_distributions(); runtime.verify_source_fingerprints(); n = runtime.verify_tokenizer_offline('/app/model'); print('pinned packages, source and tokenizer assets verified; Cangjie words:', n)"

# Record what this image really contains (the authoritative lock; compare with requirements.lock).
RUN pip freeze > /app/pip-freeze.txt

COPY ref_voice.wav /app/ref_voice.wav

ENV DEFAULT_REF_WAV=/app/ref_voice.wav

COPY handler.py .

CMD ["python", "-u", "handler.py"]
