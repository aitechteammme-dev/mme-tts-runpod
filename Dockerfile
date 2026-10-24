FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep torch/torchaudio on the CUDA 12.8 build (chatterbox may pull other versions)
RUN pip install --no-cache-dir \
    --force-reinstall \
    torch==2.7.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Bake model weights into the image so cold starts never download from Hugging Face
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox')"

# Bake the reference voice (Utkarsh clone) so it is never sent per request
COPY ref_voice.wav /app/ref_voice.wav

# Weights are local now: skip Hub network checks at startup
ENV HF_HUB_OFFLINE=1
ENV DEFAULT_REF_WAV=/app/ref_voice.wav

COPY handler.py .

CMD ["python", "-u", "handler.py"]
