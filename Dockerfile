# syntax=docker/dockerfile:1.7
#
# Athenas API — FastAPI image.
#
# Multi-stage build keeps the final image lean:
#   - builder : installs deps via `uv` (no dev group, no cache, frozen lockfile)
#   - runtime : slim Python 3.12 base + the synced site-packages + app code
#
# The image does NOT bundle `data/` (sample photos) — see `.dockerignore`.
# Mount your own lesion / scalp samples at runtime if you need to test via curl.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

# System deps needed by `uv` itself and by `pillow-heif` / `opencv-python`.
# No need for OpenCV GUI deps — `opencv-python` (the headless wheel) ships its own.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libheif-dev \
 && rm -rf /var/lib/apt/lists/*

# Install `uv` (single static binary). https://docs.astral.sh/uv/
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Copy only what `uv sync` needs to resolve the lockfile.
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY app ./app

# Production install — no dev deps, frozen lockfile, no cache to keep the
# layer small.
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --no-cache-dir uv \
 && uv sync --frozen --no-dev --no-install-project \
 && uv pip install --no-deps .

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# Runtime system deps — minimal set for OpenCV + pillow-heif.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libheif1 \
 && rm -rf /var/lib/apt/lists/*

# Non-root user — uploaded photos and tempfiles live under /app.
RUN groupadd --system --gid 1000 app \
 && useradd --system --uid 1000 --gid app --home /app --shell /sbin/nologin app

WORKDIR /app

# Copy the prepared venv and the app source from the builder stage.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/app /app/app
COPY --from=builder --chown=app:app /app/pyproject.toml /app/uv.lock ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ATHENAS_CORS_ALLOW_ORIGINS='["*"]'

USER app

EXPOSE 8000

# Cheap container-level healthcheck — no extra deps required.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1

# Single uvicorn worker is fine for MVP1 (per-request CV is run via `asyncio.to_thread`).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]