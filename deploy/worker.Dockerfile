# GPU worker: Python model stages. CUDA user-space ships inside the torch wheels,
# so the base image needs no CUDA toolkit; the host only provides the driver.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/models/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git g++ libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY worker/pyproject.toml worker/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra models
COPY worker/ ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra models

ENTRYPOINT ["python", "-m", "shu2al_worker"]
