"""Trichology orchestrator — async wraps the existing CLI ``run_pipeline``."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import UploadFile

from app.core.config import Settings
from app.services.pdf import build_trichology_pdf
from athenas.config import PipelineConfig
from athenas.pipeline import run_pipeline


async def _spool_to_tempfile(upload: UploadFile, tmp_dir: Path) -> Path:
    """Persist an UploadFile to disk; needed by :func:`athenas.io.read_rgb`."""
    suffix = Path(upload.filename or "").suffix or ".bin"
    fd, path = tempfile.mkstemp(dir=tmp_dir, suffix=suffix)
    try:
        while True:
            chunk = await upload.read(8192)
            if not chunk:
                break
            os.write(fd, chunk)
    finally:
        os.close(fd)
    return Path(path)


async def compare(
    baseline: UploadFile,
    followup: UploadFile,
    *,
    bg_method: Literal["grabcut", "rembg"] | None,
    settings: Settings,
    baseline_date: str | None = None,
    followup_date: str | None = None,
) -> bytes:
    """Compare two trichology photos and return the trichology PDF bytes.

    Optional ``baseline_date`` and ``followup_date`` (e.g. ``"2026-01-15"``)
    are appended to the per-photo legend in every figure so temporal
    context follows the photo caption.
    """
    overrides: dict[str, object] = {
        "display_height_px": settings.trichology_display_height_px,
    }
    overrides["background_method"] = bg_method or settings.trichology_bg_method
    overrides["clahe_clip_limit"] = settings.trichology_clahe_clip_limit
    config = PipelineConfig(**overrides)

    with tempfile.TemporaryDirectory(prefix="athenas_tri_") as tmp:
        tmp_dir = Path(tmp)
        path_a = await _spool_to_tempfile(baseline, tmp_dir)
        path_b = await _spool_to_tempfile(followup, tmp_dir)

        # The pipeline is fully synchronous and CPU-bound; run it on the
        # stdlib thread pool so the event loop stays responsive.
        result = await asyncio.to_thread(
            run_pipeline,
            str(path_a), str(path_b), config,
            date_a=baseline_date,
            date_b=followup_date,
        )
        return await asyncio.to_thread(
            build_trichology_pdf,
            result,
            baseline_name=baseline.filename or "A",
            followup_name=followup.filename or "B",
            baseline_date=baseline_date,
            followup_date=followup_date,
        )
