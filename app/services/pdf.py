"""PDF builders — wrap the existing trichology report and add a new dermatology one.

Both builders return raw ``bytes`` so the FastAPI layer can wrap them in a
``StreamingResponse`` without touching disk.
"""

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — headless backend, must precede pyplot.

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from athenas.report import make_report

if TYPE_CHECKING:
    # Imported only for type hints — keeps the import chain
    # `pdf → filters → service → pdf` from being evaluated at runtime
    # (otherwise ``app.services.dermatology``'s ``__init__.py`` triggers
    # loading ``service.py`` while ``pdf`` is still being constructed).
    from app.services.dermatology.filters import LesionResult

# ---------------------------------------------------------------------------
# Trichology PDF (delegates to athenas.report.make_report)
# ---------------------------------------------------------------------------


def build_trichology_pdf(
    result,
    baseline_name: str,
    followup_name: str,
    baseline_date: str | None = None,
    followup_date: str | None = None,
) -> bytes:
    """Render the existing 3-page report to bytes (no disk).

    Optional ``baseline_date`` and ``followup_date`` are forwarded to
    :func:`athenas.report.make_report` so the per-photo legends and the
    metrics page can carry the dates.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        make_report(
            result,
            tmp_path,
            baseline_name=baseline_name,
            followup_name=followup_name,
            baseline_date=baseline_date,
            followup_date=followup_date,
        )
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Dermatology PDF — NEW, A4 landscape, 1 page per lesion with a 2×2 grid
# ---------------------------------------------------------------------------


_FILTER_LABELS: list[tuple[str, str]] = [
    ("clahe", "1 · CLAHE (Lab-L)"),
    ("bw", "2 · B&W alta definición"),
    ("color", "3 · Color con mejor contraste"),
    ("normalised", "4 · Normalización de escala"),
]


def _to_3ch(arr: np.ndarray) -> np.ndarray:
    """Duplicate a gray (H×W) array into (H×W×3) for matplotlib imshow."""
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    return arr


def _resize_to_panel(arr: np.ndarray, target_h: int) -> np.ndarray:
    """Resize to a fixed panel height, preserving AR."""
    h, w = arr.shape[:2]
    if h <= target_h:
        return arr
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(arr, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _draw_filter_page(
    pdf: PdfPages,
    lesions: list[LesionResult],
    names: list[str],
    dates: list[str | None],
    *,
    filter_key: str,
    filter_label: str,
) -> None:
    """One A4-landscape page: temporal series (A vs B vs C) under one filter.

    Each lesion contributes one panel; the panel shows that lesion's
    output for ``filter_key`` (e.g. ``clahe``, ``bw``, ``color`` or
    ``normalised``) — so within one page the clinician sees the temporal
    drift of the lesion under the same filter treatment.
    """
    n = len(lesions)
    page_w, page_h = 11.0, 8.5  # A4 landscape in inches
    fig, axs = plt.subplots(1, n, figsize=(page_w, page_h))
    if n == 1:
        axs = [axs]

    suptitle = f"Athenas — Dermatología · {filter_label}"
    fig.suptitle(suptitle, fontsize=14, weight="bold", y=0.985)

    for i, (ax, lesion, name, date) in enumerate(
        zip(axs, lesions, names, dates, strict=True), start=1,
    ):
        rgb = getattr(lesion.filters, filter_key)
        rgb = _to_3ch(rgb)
        rgb = _resize_to_panel(rgb, target_h=420)
        ax.imshow(rgb)
        caption = f"Lesión {i} · {name}"
        if date:
            caption += f"\nFecha: {date}"
        ax.set_xlabel(caption, fontsize=11, weight="bold", labelpad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def _draw_notes_page(
    pdf: PdfPages,
    lesion_names: list[str],
    lesion_dates: list[str | None],
) -> None:
    """Final page documenting what each filter does + clinical disclaimer."""
    page_w, page_h = 8.5, 11.0  # back to portrait for the text-heavy page
    fig, ax = plt.subplots(figsize=(page_w, page_h))
    ax.axis("off")

    fig.suptitle(
        "Athenas — Dermatología · Notas",
        fontsize=13, weight="bold", y=0.975,
    )

    cursor_y = 0.95
    line_height = 0.022

    def line_(text: str, *, bold: bool = False, italic: bool = False, size: float = 10.0) -> None:
        nonlocal cursor_y
        ax.text(
            0.05, cursor_y, text,
            ha="left", va="top", transform=ax.transAxes,
            fontsize=size, weight="bold" if bold else "normal",
            style="italic" if italic else "normal",
        )
        cursor_y -= line_height

    line_("Disclaimer clínico", bold=True)
    line_(
        "MVP visual-aid. Las figuras son descriptores de las fotografías — "
        "NO un diagnóstico clínico ni un conteo automatizado de folículos. "
        "Un dermatólogo entrenado debe revisar las imágenes antes de "
        "extraer cualquier conclusión.",
        italic=True, size=9.5,
    )
    cursor_y -= 0.02

    line_("Lesiones comparadas", bold=True)
    for i, (name, date) in enumerate(zip(lesion_names, lesion_dates, strict=True), start=1):
        line_text = f"  · Lesión {i}: {name}"
        if date:
            line_text += f"  ·  {date}"
        line_(line_text, size=9.5)
    cursor_y -= 0.02

    line_("Estructura del reporte", bold=True)
    line_(
        "Cada filtro visual (CLAHE, B&W alta definición, Color con mejor "
        "contraste, Normalización de escala) tiene su propia página, con "
        "la serie temporal lado a lado (A vs B vs C). Esto permite ver "
        "cómo evoluciona la lesión bajo el mismo tratamiento de realce.",
        italic=True, size=9.5,
    )
    cursor_y -= 0.02

    notes = [
        (
            "CLAHE (Lab-L)",
            "CLAHE sobre la L de Lab. Conserva el color (a*, b* intactos). "
            "Realza contraste local de luminancia.",
        ),
        (
            "B&W alta definición",
            "Gris + CLAHE + unsharp mask. Resalta bordes de lesión y "
            "asimetrías de borde.",
        ),
        (
            "Color con mejor contraste",
            "Lab-L CLAHE + HSV-S × 1.2. Boost combinado de luminancia y "
            "chroma — hace que eritema y pigmentación sean más obvios.",
        ),
        (
            "Normalización de escala",
            "Resize a altura fija para que las fotos temporales se vean a "
            "tamaño consistente — facilita la métrica visual de crecimiento.",
        ),
    ]
    for heading, body in notes:
        line_(heading, bold=True)
        line_(body, italic=True, size=9.5)
        cursor_y -= 0.015

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


def build_dermatology_pdf(
    lesions: list[LesionResult],
    names: list[str],
    dates: list[str | None] | None = None,
) -> bytes:
    """Render the dermatology PDF to bytes (no disk).

    Layout (always ``len(_FILTER_LABELS) + 1`` pages, regardless of N photos):

    * One page per filter — temporal series side-by-side (A vs B vs C
      within that filter).
    * Final notes page — clinical disclaimer + per-filter explanation.

    ``dates`` is an optional list of human-readable date strings, parallel
    to ``names`` (same length as ``lesions``). Missing entries are
    silently skipped — both the per-page captions and the final notes
    page omit them.
    """
    if dates is None:
        dates = [None] * len(lesions)
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        for filter_key, filter_label in _FILTER_LABELS:
            _draw_filter_page(
                pdf, lesions, names, dates,
                filter_key=filter_key,
                filter_label=filter_label,
            )
        _draw_notes_page(pdf, names, dates)
    return buf.getvalue()
