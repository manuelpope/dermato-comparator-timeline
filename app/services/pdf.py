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
    from app.services.aesthetics.analysis import AestheticsAnalysis
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


# ---------------------------------------------------------------------------
# Aesthetics PDF — NEW, 3 pages: normalised + wrinkle relief, B&W + change
# map, and the metrics / protocol-check table.
# ---------------------------------------------------------------------------


def _aesthetics_caption(
    label: str,
    date: str | None,
    name: str | None,
) -> str:
    """Caption under one face panel: 'Antes · 2026-01-15 · file.png'."""
    parts = [label]
    if date:
        parts.append(date)
    if name:
        parts.append(name)
    return " · ".join(parts)


def _draw_aesthetics_faces_page(
    pdf: PdfPages,
    analysis: AestheticsAnalysis,
    before_caption: str,
    after_caption: str,
) -> None:
    """Page 1 — the normalised pair + the expression-line relief."""
    views = analysis.views
    page_w, page_h = 11.0, 8.5  # A4 landscape
    fig, axs = plt.subplots(2, 2, figsize=(page_w, page_h))
    fig.suptitle(
        "Athenas — Estética · Rostro normalizado y líneas de expresión",
        fontsize=14, weight="bold", y=0.98,
    )

    panels = [
        (axs[0][0], views.before.normalised, before_caption),
        (axs[0][1], views.after.normalised, after_caption),
        (axs[1][0], views.before.wrinkles, f"{before_caption} · líneas"),
        (axs[1][1], views.after.wrinkles, f"{after_caption} · líneas"),
    ]
    for ax, image, caption in panels:
        ax.imshow(_to_3ch(image))
        ax.set_xlabel(caption, fontsize=10, weight="bold", labelpad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(
        0.5, 0.015,
        "Ambas caras están alineadas, recortadas y escaladas al mismo tamaño "
        "de cabeza (mismo eje de ojos) para que la comparación sea válida.",
        ha="center", fontsize=8.5, style="italic", color="#444444",
    )
    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def _draw_aesthetics_bw_change_page(
    pdf: PdfPages,
    analysis: AestheticsAnalysis,
    before_caption: str,
    after_caption: str,
) -> None:
    """Page 2 — B&W relief (both photos) + the before→after change map."""
    views = analysis.views
    page_w, page_h = 11.0, 8.5
    fig = plt.figure(figsize=(page_w, page_h))
    fig.suptitle(
        "Athenas — Estética · Relieve en B&W (CLAHE + nitidez) y mapa de cambio",
        fontsize=14, weight="bold", y=0.98,
    )
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])

    ax_bw_before = fig.add_subplot(grid[0, 0])
    ax_bw_after = fig.add_subplot(grid[0, 1])
    ax_change = fig.add_subplot(grid[1, :])

    ax_bw_before.imshow(_to_3ch(views.before.bw))
    ax_bw_before.set_xlabel(before_caption, fontsize=10.5, weight="bold")
    ax_bw_after.imshow(_to_3ch(views.after.bw))
    ax_bw_after.set_xlabel(after_caption, fontsize=10.5, weight="bold")
    ax_change.imshow(views.change_map)
    ax_change.set_xlabel(
        "Mapa de cambio antes → después (INFERNO: rojo = mayor diferencia)",
        fontsize=10.5, weight="bold",
    )

    for ax in (ax_bw_before, ax_bw_after, ax_change):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def _draw_aesthetics_metrics_page(
    pdf: PdfPages,
    analysis: AestheticsAnalysis,
    *,
    date_before: str | None,
    date_after: str | None,
    before_name: str | None,
    after_name: str | None,
    patient_id: str | None = None,
) -> None:
    """Page 3 — per-region densities + protocol checks + disclaimer."""
    page_w, page_h = 8.5, 11.0  # portrait
    fig, ax = plt.subplots(figsize=(page_w, page_h))
    ax.axis("off")
    title = "Athenas — Estética · Métricas y protocolo de captura"
    if patient_id:
        title = f"Athenas — Estética · {patient_id} · Métricas y protocolo de captura"
    fig.suptitle(title, fontsize=13, weight="bold", y=0.975)

    cursor_y = 0.945
    line_height = 0.026

    def line_(text: str, *, bold: bool = False, italic: bool = False, size: float = 11.5) -> None:
        nonlocal cursor_y
        ax.text(
            0.05, cursor_y, text,
            ha="left", va="top", transform=ax.transAxes,
            fontsize=size, weight="bold" if bold else "normal",
            style="italic" if italic else "normal",
        )
        cursor_y -= line_height

    line_("Comparación", bold=True)
    line_(f"Foto anterior: {date_before or '—'} · {before_name or '—'}", size=11)
    line_(f"Foto posterior: {date_after or '—'} · {after_name or '—'}", size=11)
    cursor_y -= 0.012

    alignment = analysis.alignment
    line_("Normalización aplicada", bold=True)
    line_(
        f"Escala corregida ×{alignment.scale:.3f} · "
        f"rotación {alignment.rotation_deg:+.2f}° · "
        f"distancia interpupilar relativa {alignment.ipd_ratio_b_over_a:.3f}",
        size=11,
    )
    line_(
        f"Ambas caras se recortaron con la misma ventana y se escalaron a "
        f"{analysis.target_height} px de alto.",
        italic=True, size=11,
    )
    cursor_y -= 0.012

    line_("Densidad de líneas por región", bold=True)
    headers = ["Región", "Antes", "Después", "Cambio"]
    rows = []
    for delta in analysis.region_deltas:
        if delta.delta_pct is None:
            change = "—"
        elif delta.delta_pct < -1:
            change = f"▼ {abs(delta.delta_pct):.1f}%"
        elif delta.delta_pct > 1:
            change = f"▲ {delta.delta_pct:.1f}%"
        else:
            change = f"{delta.delta_pct:+.1f}%"
        rows.append(
            [
                delta.label,
                f"{delta.density_before * 100:.2f}%",
                f"{delta.density_after * 100:.2f}%",
                change,
            ]
        )
    row_h = 0.030
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="upper left",
        bbox=[0.05, cursor_y - 0.014 - row_h * len(rows), 0.9, row_h * len(rows)],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    cursor_y -= 0.014 + row_h * len(rows)
    cursor_y -= 0.012

    line_("Protocolo de captura", bold=True)
    for check in analysis.protocol_checks:
        icon = "OK" if check.passed else "REVISAR"
        line_(f"[{icon}] {check.message}", size=11)
    cursor_y -= 0.012

    line_("Disclaimer clínico", bold=True)
    line_(
        "Ayuda visual para la comparación antes/después de una sesión de "
        "botox. Las densidades de línea son un descriptor cuantitativo de la "
        "imagen, no un diagnóstico ni una medición clínica validada. La "
        "valoración final es del profesional tratante.",
        italic=True, size=11,
    )

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


def build_aesthetics_pdf(
    analysis: AestheticsAnalysis,
    *,
    patient_id: str | None = None,
    date_before: str | None = None,
    date_after: str | None = None,
    before_name: str | None = None,
    after_name: str | None = None,
) -> bytes:
    """Render the aesthetics (Botox before/after) PDF to bytes (no disk).

    Three pages:

    * Page 1 — the normalised pair (colour) + the expression-line relief
      (magenta overlay) of both photos.
    * Page 2 — the B&W high-definition pair + the before→after change map.
    * Page 3 — per-region density table, capture-protocol checks and the
      clinical disclaimer.
    """
    before_caption = _aesthetics_caption("Antes", date_before, before_name)
    after_caption = _aesthetics_caption("Después", date_after, after_name)
    before_short = _aesthetics_caption("Antes", date_before, None)
    after_short = _aesthetics_caption("Después", date_after, None)
    before_title = f"{patient_id} · {before_caption}" if patient_id else before_caption
    after_title = f"{patient_id} · {after_caption}" if patient_id else after_caption

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        _draw_aesthetics_faces_page(pdf, analysis, before_title, after_title)
        _draw_aesthetics_bw_change_page(pdf, analysis, before_short, after_short)
        _draw_aesthetics_metrics_page(
            pdf,
            analysis,
            date_before=date_before,
            date_after=date_after,
            before_name=before_name,
            after_name=after_name,
            patient_id=patient_id,
        )
    return buf.getvalue()
