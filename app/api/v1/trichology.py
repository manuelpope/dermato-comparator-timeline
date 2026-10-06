"""Trichology endpoint — POST /v1/trichology/compare."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.services import trichology as service

router = APIRouter(tags=["trichology"])


@router.post("/trichology/compare")
async def compare(
    baseline: Annotated[UploadFile, File(description="Baseline trichology photo (A).")],
    followup: Annotated[UploadFile, File(description="Follow-up trichology photo (B).")],
    bg_method: Annotated[
        Literal["grabcut", "rembg"] | None,
        Form(description="Background-removal backend. Default = settings default."),
    ] = None,
    patient_id: Annotated[
        str | None,
        Form(description="Patient ID or reference; baked into the PDF name."),
    ] = None,
    baseline_date: Annotated[
        str | None,
        Form(description="Human-readable date for baseline (e.g. '2026-01-15')."),
    ] = None,
    followup_date: Annotated[
        str | None,
        Form(description="Human-readable date for follow-up (e.g. '2026-07-11')."),
    ] = None,
) -> StreamingResponse:
    """Compare two trichology photos and return a single PDF."""
    settings = get_settings()
    try:
        pdf_bytes = await service.compare(
            baseline=baseline,
            followup=followup,
            bg_method=bg_method,
            settings=settings,
            baseline_date=baseline_date,
            followup_date=followup_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    today = datetime.now().strftime("%Y%m%d")
    patient_slug = patient_id or "patient"
    a = Path(baseline.filename or "A").stem
    b = Path(followup.filename or "B").stem
    filename = f"trichology_{patient_slug}_{today}_{a}_vs_{b}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
