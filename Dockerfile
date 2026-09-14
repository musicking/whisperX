# CUDA/cuDNN are provided by the image; the host only needs the NVIDIA driver
# and NVIDIA Container Toolkit.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg libgomp1 libpython3.12t64 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    UV_LINK_MODE=copy \
    HF_HUB_CACHE=/data/models/hf-hub \
    TORCH_HOME=/data/models/torch \
    NLTK_DATA=/data/models/nltk-data:/opt/nltk_data

# PyTorch installs its own cuDNN/cuBLAS. Use those libraries for CTranslate2 too.
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

COPY pyproject.toml uv.lock README.md LICENSE MANIFEST.in ./
COPY whisperx ./whisperx
# Isolated Python checks the installed wheel, not the source under /app.
RUN uv sync --frozen --extra api --python /usr/bin/python3 --no-editable \
    && python -I -c "from importlib.resources import files; assets = files('whisperx').joinpath('assets'); assert all(assets.joinpath(name).is_file() for name in ('pytorch_model.bin', 'mel_filters.npz')), 'WhisperX package assets are missing'" \
    && python -m nltk.downloader -d /opt/nltk_data punkt_tab \
    && uv cache clean

EXPOSE 7865
CMD ["whisperx-api"]
