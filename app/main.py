"""FastAPI application factory for the Athenas ecosystem (MVP1)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import router as v1_router
from app.core.config import get_settings

log = logging.getLogger("athenas.api")

# The guided UI (static/index.html) lives next to the ``app`` package, so it
# resolves the same way in a checkout, in the Docker image and in tests.
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """MVP1: stateless. Just log startup / shutdown.

    Future iterations can pre-load rembg models here, warm a thread pool,
    or open a DB connection. Keeping it lean keeps cold-start fast.
    """
    log.info("Athenas API ready (MVP1, stateless).")
    yield
    log.info("Athenas API shutting down.")


def create_app() -> FastAPI:
    """Application factory — easy to test with ``create_app()`` directly."""
    settings = get_settings()
    app = FastAPI(
        title="Athenas API",
        version="0.2.0",
        description=(
            "Visual-aid endpoints for trichology and dermatology follow-up "
            "photographs. MVP1 — no auth, no storage."
        ),
        lifespan=lifespan,
    )

    # Permissive CORS for dev — tighten to a strict allowlist in MVP2.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # The guide reads the PDF filename off the response header, which is
        # unreadable cross-origin unless it is explicitly exposed.
        expose_headers=["Content-Disposition"],
    )

    app.include_router(v1_router, prefix="/v1")

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": app.version}

    # The guided UI is mounted last so it never shadows /health, /docs or /v1.
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="guide")

    return app


app = create_app()
