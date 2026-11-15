# syntax=docker/dockerfile:1
# ==============================================================================
# TensorMMU Container Image (Linux x86_64 / ARM64 + NVIDIA Ampere/Ada/Hopper/Blackwell sm_80+)
# - Includes CUDA 12.4 NVRTC runtime headers for JIT-compiled MoE-TLB & Zero-Copy kernels
# - Starts HTTP server immediately:
#     * GET  /livez          -> 200 OK immediately
#     * GET  /readyz         -> 503 while downloading/compiling/loading, 200 OK when ready
#     * POST /v1/messages    -> 503 (Retry-After: 10) until ready, then serves Anthropic SSE/JSON
#     * POST /v1/chat/completions -> 503 until ready, then serves OpenAI SSE/JSON
# ==============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_MODULE_LOADING=LAZY \
    HF_HOME=/models/hf_cache \
    TMMU_CHECKPOINTS_DIR=/models/cartridges \
    TMMU_HOST=0.0.0.0 \
    TMMU_PORT=8000 \
    TMMU_DEFAULT_MAX_TOKENS=32768 \
    TMMU_THINKING_BUDGET=6144

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        curl \
        git \
        ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CUDA 12.4 wheels first for optimal Docker layer caching
RUN pip install --index-url https://download.pytorch.org/whl/cu124 torch>=2.4.0

WORKDIR /app
COPY requirements.txt pyproject.toml README.md LICENSE ./
RUN pip install -r requirements.txt

COPY tensormmu ./tensormmu
COPY runtime ./runtime
COPY decompiler ./decompiler
COPY data ./data
COPY eval ./eval
COPY models ./models
COPY docs ./docs
COPY serve_nvm_openai.py ./

RUN pip install --no-build-isolation --no-deps -e . \
    && mkdir -p /models/hf_cache /models/cartridges

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/livez || exit 1

ENTRYPOINT ["tmmu", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
