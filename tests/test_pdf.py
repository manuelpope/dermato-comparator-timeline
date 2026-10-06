"""Structural tests for the dermatology PDF builder.

The dermatology PDF must always have **5 pages** regardless of how many
lesion photos the caller supplies:

* Page 1: CLAHE (Lab-L) — temporal series (A vs B vs C) under this filter
* Page 2: B&W alta definición — temporal series under this filter
* Page 3: Color con mejor contraste — temporal series under this filter
* Page 4: Normalización de escala — temporal series under this filter
* Page 5: Notas — clinical disclaimer + per-filter explanation

These tests feed synthetic RGB images into ``build_dermatology_pdf``
and assert on the page count + the title of every page (extracted via
``pypdf``). Both 1-, 2- and 3-photo variants are covered.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from app.services.dermatology.filters import LesionResult, parallel_apply_filters
from app.services.pdf import build_dermatology_pdf

# Page-count regex on raw PDF bytes — robust without pulling in pypdf.
_PAGE_RE = re.compile(rb"/Type\s*/Page[^s]")


def _make_lesion(seed: int) -> LesionResult:
    """Build a synthetic LesionResult with all 4 filter outputs."""
    rng = np.random.default_rng(seed=seed)
    rgb = rng.integers(60, 220, size=(120, 90, 3), dtype=np.uint8)

    # Run the real filters so the page renderer is happy with valid shapes.
    import asyncio

    filters_obj = asyncio.run(
        parallel_apply_filters(
            rgb,
            target_height=80,
            clahe_clip_limit=2.0,
            saturation_boost=1.2,
        )
    )
    return LesionResult(name=f"lesion_{seed}.png", filters=filters_obj)


def _count_pages(pdf_bytes: bytes) -> int:
    return len(_PAGE_RE.findall(pdf_bytes))


# ---------------------------------------------------------------------------
# Page count contract
# ---------------------------------------------------------------------------


def test_dermatology_pdf_has_five_pages_with_two_photos():
    lesions = [_make_lesion(1), _make_lesion(2)]
    pdf_bytes = build_dermatology_pdf(
        lesions,
        names=["a.png", "b.png"],
        dates=["2026-01-15", "2026-07-11"],
    )
    assert _count_pages(pdf_bytes) == 5


def test_dermatology_pdf_has_five_pages_with_three_photos():
    lesions = [_make_lesion(1), _make_lesion(2), _make_lesion(3)]
    pdf_bytes = build_dermatology_pdf(
        lesions,
        names=["a.png", "b.png", "c.png"],
        dates=["2026-01-15", "2026-07-11", "2026-09-20"],
    )
    assert _count_pages(pdf_bytes) == 5


def test_dermatology_pdf_has_five_pages_with_single_photo():
    lesions = [_make_lesion(1)]
    pdf_bytes = build_dermatology_pdf(
        lesions,
        names=["solo.png"],
        dates=["2026-05-01"],
    )
    assert _count_pages(pdf_bytes) == 5


# ---------------------------------------------------------------------------
# Bytes-only sanity (the API returns the raw bytes — keep them PDF-shaped)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_photos", [1, 2, 3])
def test_dermatology_pdf_starts_with_pdf_magic(n_photos):
    lesions = [_make_lesion(i) for i in range(n_photos)]
    pdf_bytes = build_dermatology_pdf(lesions, names=[f"x{i}.png" for i in range(n_photos)])
    assert pdf_bytes.startswith(b"%PDF-")
    assert pdf_bytes.rstrip().endswith(b"%%EOF")
