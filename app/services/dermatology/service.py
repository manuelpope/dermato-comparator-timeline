"""Dermatology orchestrator — load N images, run 4 filters in parallel, emit PDF.

The 4 visual filters (CLAHE, B&W contrast, color contrast, scale
normalisation) are applied **only to the most recent** photo (the last
upload in the series — ``photo_c`` if 3 were given, else ``photo_b``,
else ``photo_a``). Older photos are kept as raw RGB for the temporal
evolution page so the clinician can see growth/colour drift side by
side with the filtered current state.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from fastapi import UploadFile

from app.core.config import Settings
from app.services.dermatology.filters import parallel_apply_filters
from app.services.pdf import build_dermatology_pdf
from athenas.io import read_rgb


async def _spool_to_tempfile(upload: UploadFile, tmp_dir: Path) -> Path:
    """Persist an UploadFile's bytes into a real on-disk file.

    We can't feed :func:`read_rgb` (which expects a path or buffer) with a
    raw bytes object cleanly across OpenCV/Pillow, and the HEIC plugin is
    registered on import of :mod:`athenas.io` — so writing to a real
    file inside the request-scoped tempdir is the simplest, safest path.
    """
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


async def _read_one(upload: UploadFile, tmp_dir: Path) -> np.ndarray:
    """Read one upload into an RGB ndarray on the stdlib thread pool."""
    path = await _spool_to_tempfile(upload, tmp_dir)
    return await asyncio.to_thread(read_rgb, str(path))


async def compare(
    photos: Iterable[tuple[UploadFile, str | None]],
    *,
    settings: Settings,
) -> bytes:
    """Compare 1–3 lesion photos and return the dermatology PDF bytes.

    Each entry in ``photos`` is ``(upload, date_str_or_None)``. The
    length determines the report size (2 entries by default, 3 when the
    caller supplies ``photo_c``).
    """
    photos_list = list(photos)
    if not 1 <= len(photos_list) <= 3:
        raise ValueError(
            f"Dermatology compare accepts 1..3 photos, got {len(photos_list)}."
        )

    with tempfile.TemporaryDirectory(prefix="athenas_derm_") as tmp:
        tmp_dir = Path(tmp)

        # 1. Read all uploads in parallel (independent I/O).
        uploads = [u for u, _ in photos_list]
        rgbs = await asyncio.gather(*(_read_one(u, tmp_dir) for u in uploads))

        # 2. Run all 4 filters on all lesions in parallel.
        per_image = await asyncio.gather(
            *(
                parallel_apply_filters(
                    rgb,
                    target_height=settings.derm_target_height_px,
                    clahe_clip_limit=settings.derm_clahe_clip_limit,
                    saturation_boost=settings.derm_saturation_boost,
                )
                for rgb in rgbs
            )
        )

        # 3. Bundle into LesionResults, one per photo (with its date).
        lesions = [
            LesionResult(
                name=upload.filename or f"Lesión {i + 1}",
                filters=filters,
            )
            for i, ((upload, _date), filters) in enumerate(
                zip(photos_list, per_image, strict=True),
            )
        ]
        names = [lesion.name for lesion in lesions]
        dates = [date for _, date in photos_list]

        # 4. Build the PDF on the stdlib pool (matplotlib is synchronous).
        return await asyncio.to_thread(
            build_dermatology_pdf, lesions, names, dates,
        )
