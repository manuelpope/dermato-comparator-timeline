"""Dermatology endpoint — POST /v1/dermatology/compare.

Simple, explicit upload: ``photo_a`` + ``photo_b`` are required (default
2-photo case), ``photo_c`` is optional for the 3-photo case. Each photo
has an optional date string (``date_a``, ``date_b``, ``date_c``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.services import dermatology as service

router = APIRouter(tags=["dermatology"])


@router.post("/dermatology/compare")
async def compare(
    photo_a: Annotated[UploadFile, File(description="Lesion photo A.")],
    photo_b: Annotated[UploadFile, File(description="Lesion photo B.")],
    photo_c: Annotated[
        UploadFile | None,
        File(description="Optional third lesion photo (C)."),
    ] = None,
    date_a: Annotated[
        str | None,
        Form(description="Human-readable date for photo A (e.g. '2026-01-15')."),
    ] = None,
    date_b: Annotated[
        str | None,
        Form(description="Human-readable date for photo B (e.g. '2026-03-15')."),
    ] = None,
    date_c: Annotated[
        str | None,
        Form(description="Human-readable date for photo C (e.g. '2026-09-20')."),
    ] = None,
    target_height: Annotated[
        int,
        Form(
            ge=200,
            le=2000,
            description="Resize normalised lesion to this height in px (default 600).",
        ),
    ] = 600,
    clahe_clip_limit: Annotated[
        float,
        Form(
            ge=0.5,
            le=10.0,
            description="CLAHE clip limit for the luminance + color filters (default 2.0).",
        ),
    ] = 2.0,
    saturation_boost: Annotated[
        float,
        Form(
            ge=1.0,
            le=2.0,
            description="HSV-S × this factor for the color-contrast filter (default 1.2).",
        ),
    ] = 1.2,
    patient_id: Annotated[
        str | None,
        Form(description="Patient ID or reference; baked into the PDF name."),
    ] = None,
) -> StreamingResponse:
    """Compare 2 (or 3) skin-lesion photos and return the dermatology PDF."""
    settings = get_settings()

    photos: list[tuple[UploadFile, str | None]] = [(photo_a, date_a), (photo_b, date_b)]
    if photo_c is not None:
        photos.append((photo_c, date_c))

    try:
        pdf_bytes = await service.compare(photos=photos, settings=settings)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    today = datetime.now().strftime("%Y%m%d")
    patient_slug = patient_id or "patient"
    filename = f"dermatology_{patient_slug}_{today}_{len(photos)}_lesions.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
