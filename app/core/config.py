"""Centralised settings for the Athenas FastAPI service.

Read from environment variables prefixed with ``ATHENAS_`` (e.g.
``ATHENAS_TRICHOLOGY_BG_METHOD=rembg``) or from a local ``.env`` file.

Keep this surface minimal — every field here is something a power user
might want to override in production.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration. Cached via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="ATHENAS_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- HTTP / CORS (MVP1: open for local dev) -------------------------
    # MVP2 will tighten this to a strict allowlist once the API is exposed.
    cors_allow_origins: list[str] = ["*"]
    max_upload_size_mb: int = 25  # per-file cap

    # ---- Trichology pass-through defaults -------------------------------
    # We default to ``grabcut`` in the API even though the CLI prefers
    # ``rembg`` — rembg downloads ~200 MB on first call and the API should
    # fail fast on a fresh process. Clients can opt in per-request.
    trichology_bg_method: Literal["grabcut", "rembg"] = "grabcut"
    trichology_display_height_px: int = 800
    trichology_clahe_clip_limit: float = 3.0

    # ---- Dermatology defaults -------------------------------------------
    derm_target_height_px: int = 600
    derm_clahe_clip_limit: float = 2.0
    derm_saturation_boost: float = 1.2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
