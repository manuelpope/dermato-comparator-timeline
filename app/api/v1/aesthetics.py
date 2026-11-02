"""Aesthetics endpoint — POST /v1/aesthetics/compare.

Compares a "before" and an "after" Botox photo of the same patient. The
two photos are normalised (aligned, cropped and scaled so the heads have
the same size and the eyes sit on the same line) and then rendered side
by side under four treatments: the normalised face, an expression-line
relief overlay, a B&W high-definition view and a before→after change map.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.v1._headers import safe_filename_part
from app.core.config import get_settings
from app.services import aesthetics as service

router = APIRouter(tags=["aesthetics"])


def _slug_date(value: str | None) -> str:
    """Turn a human date ('2026-01-15') into a filename-safe token."""
    if not value:
        return datetime.now().strftime("%Y%m%d")
    cleaned = "".join(ch for ch in value if ch.isalnum())
    return cleaned or datetime.now().strftime("%Y%m%d")


@router.post("/aesthetics/compare")
async def compare(
    photo_before: Annotated[
        UploadFile,
        File(description="Foto del rostro ANTES de la sesión de botox."),
    ],
    photo_after: Annotated[
        UploadFile,
        File(description="Foto del rostro DESPUÉS de la sesión de botox."),
    ],
    date_before: Annotated[
        str | None,
        Form(description="Fecha de la foto anterior (p. ej. '2026-01-15')."),
    ] = None,
    date_after: Annotated[
        str | None,
        Form(description="Fecha de la foto posterior (p. ej. '2026-03-02')."),
    ] = None,
    patient_id: Annotated[
        str | None,
        Form(description="ID o referencia del paciente; se incluye en el PDF."),
    ] = None,
    target_height: Annotated[
        int | None,
        Form(
            ge=200,
            le=2000,
            description=(
                "Alto en px de la cara normalizada (por defecto, el valor de "
                "ATHENAS_AESTHETICS_TARGET_HEIGHT_PX)."
            ),
        ),
    ] = None,
    wrinkle_threshold: Annotated[
        float | None,
        Form(
            ge=50.0,
            le=99.5,
            description=(
                "Percentil del detector de líneas (más alto = sólo las líneas "
                "más marcadas)."
            ),
        ),
    ] = None,
) -> StreamingResponse:
    """Compare a before/after Botox pair and return the aesthetics PDF."""
    settings = get_settings()

    # The raw stems also become the PDF captions — those keep the original
    # name, while the download name below is sanitised.
    before_stem = Path(photo_before.filename or "antes").stem
    after_stem = Path(photo_after.filename or "despues").stem

    try:
        pdf_bytes = await service.compare(
            photo_before,
            photo_after,
            date_before=date_before,
            date_after=date_after,
            patient_id=patient_id,
            before_name=before_stem,
            after_name=after_stem,
            target_height=target_height,
            wrinkle_threshold=wrinkle_threshold,
            settings=settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    patient_slug = safe_filename_part(patient_id or "patient")
    filename = (
        f"aesthetics_{patient_slug}_{_slug_date(date_before)}"
        f"_{safe_filename_part(before_stem)}_vs_{safe_filename_part(after_stem)}.pdf"
    )
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
