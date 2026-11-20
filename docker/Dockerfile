# Generic CUDA 12.8 image.
#   docker build -f docker/Dockerfile -t online-mistake-detection:cuda12 .
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_SPEC=torch>=2.10.0,<2.11.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel
RUN pip install "${TORCH_SPEC}" torchvision --index-url "${TORCH_INDEX_URL}"
RUN pip install unsloth==2026.6.7 unsloth_zoo==2026.6.5 --no-deps

COPY requirements.txt ./
COPY pyproject.toml ./
COPY src ./src
COPY docker/entrypoint.sh /usr/local/bin/online-mistake-detection-entrypoint

RUN pip install -r requirements.txt \
    && pip install -e . --no-deps \
    && chmod +x /usr/local/bin/online-mistake-detection-entrypoint

ENTRYPOINT ["online-mistake-detection-entrypoint"]
CMD ["bash"]
