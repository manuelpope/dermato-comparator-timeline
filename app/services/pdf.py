"""PDF builders — wrap the existing trichology report and add a new dermatology one.

Both builders return raw ``bytes`` so the FastAPI layer can wrap them in a
``StreamingResponse`` without touching disk.
"""

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — headless backend, must precede pyplot.

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from app.services.dermatology.filters import LesionResult
from athenas.report import make_report

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


def _draw_lesion_page(
    pdf: PdfPages,
    lesion: LesionResult,
    index: int,
    date: str | None = None,
) -> None:
    """One A4-landscape page with a 2×2 grid of the 4 filter outputs."""
    page_w, page_h = 11.0, 8.5  # A4 landscape in inches
    fig, axs = plt.subplots(2, 2, figsize=(page_w, page_h))

    suptitle = f"Athenas — Dermatología · Lesión {index}"
    if lesion.name:
        suptitle += f"  ·  {lesion.name}"
    fig.suptitle(suptitle, fontsize=13, weight="bold", y=0.995)

    # Date subtitle sits just below the title, italic, small.
    if date:
        fig.text(
            0.5, 0.965,
            f"Fecha: {date}",
            ha="center", va="top", style="italic", fontsize=10, color="#444",
        )

    sources: dict[str, np.ndarray] = {
        "clahe": lesion.filters.clahe,
        "bw": lesion.filters.bw,
        "color": lesion.filters.color,
        "normalised": lesion.filters.normalised,
    }

    for ax, (key, label) in zip(axs.flat, _FILTER_LABELS, strict=True):
        rgb = _to_3ch(sources[key])
        rgb = _resize_to_panel(rgb, target_h=320)
        ax.imshow(rgb)
        ax.set_title(label, fontsize=11, weight="bold", loc="left")
        ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2)
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

    ``dates`` is an optional list of human-readable date strings, parallel
    to ``names`` (same length as ``lesions``). Missing entries are
    silently skipped — both the per-lesion page subtitle and the final
    notes page omit them.
    """
    if dates is None:
        dates = [None] * len(lesions)
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        for idx, (lesion, date) in enumerate(zip(lesions, dates, strict=True), start=1):
            _draw_lesion_page(pdf, lesion, idx, date=date)
        _draw_notes_page(pdf, names, dates)
    return buf.getvalue()
