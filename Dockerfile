# syntax=docker/dockerfile:1
# PyTorch's pinned cu128 wheels supply CUDA 12.8 and cuDNN. Installing another
# CUDA runtime in the base image would duplicate these libraries.
FROM ubuntu:24.04 AS runtime-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg libgomp1 libpython3.12t64 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.8" \
    HF_HUB_CACHE=/data/models/hf-hub \
    TORCH_HOME=/data/models/torch \
    NLTK_DATA=/data/models/nltk-data:/opt/nltk_data

# CTranslate2 and PyTorch must load the same cuDNN/cuBLAS libraries.
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

FROM runtime-base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=5 \
    UV_CONCURRENT_DOWNLOADS=4

# Keep large dependencies cached independently of application source changes.
# BuildKit retains completed downloads after a failed build, outside the image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra api --no-dev --no-install-project --python /usr/bin/python3 \
    && python -m nltk.downloader -d /opt/nltk_data punkt_tab

COPY README.md LICENSE MANIFEST.in ./
COPY whisperx ./whisperx
# Explicit local installs always rebuild the wheel. uv sync may otherwise reuse
# a wheel when Python sources change but pyproject.toml stays unchanged.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python --no-deps . \
    && python -I -c "from pathlib import Path; from importlib.resources import files; source = Path('whisperx'); package = files('whisperx'); assert all(package.joinpath(*path.relative_to(source).parts).read_bytes() == path.read_bytes() for path in source.rglob('*.py')), 'Installed WhisperX differs from source'"

FROM runtime-base AS runtime
# Only installed packages and NLTK data enter the final image, not build caches
# or local model directories. Whisper/alignment models live on the /data mount.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/nltk_data /opt/nltk_data
# Check the installed wheel, including WhisperX's small bundled VAD asset.
RUN python -I -c "from importlib.resources import files; assets = files('whisperx').joinpath('assets'); assert all(assets.joinpath(name).is_file() for name in ('pytorch_model.bin', 'mel_filters.npz')), 'WhisperX package assets are missing'"

EXPOSE 7865
CMD ["whisperx-api"]
