FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# v3: where the model weights are baked (see below), so a cold worker never downloads them
ENV HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    --force-reinstall \
    torch==2.7.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# v3: bake the model weights into the image so cold starts never download from Hugging Face
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox')"

# v3: bake the reference voice (it becomes the voice named "utkarsh"); it is never sent per request. ref_voice.wav must be in the repo root.
COPY ref_voice.wav /app/ref_voice.wav

# v3: the weights are local now: skip Hub network checks at start-up
ENV HF_HUB_OFFLINE=1
ENV DEFAULT_REF_WAV=/app/ref_voice.wav

COPY handler.py .

CMD ["python", "-u", "handler.py"]
