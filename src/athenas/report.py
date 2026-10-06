"""Minimalist PDF report — composes 4 in-memory comparison figures into a 3-page PDF.

Layout (A4 portrait, no files on disk):
  Page 1: 1 · Original  +  2 · Sin fondo (cabeza normalizada)
  Page 2: 3 · B&W alta definición  +  4 · Cluster Lab (k=3)
  Page 3: Tabla de métricas — qué mide cada métrica y nota explicativa.

Each comparison occupies **half a page**. The 4 ``Figure`` objects coming out
of the pipeline are rasterised straight to the PDF via ``PdfPages`` — no
intermediate PNGs are ever written.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from .comparison import fig_to_rgb

if TYPE_CHECKING:
    from .pipeline import PipelineResult


# Metrics taken from each comparison figure. Spanish names (matching the
# figure labels) + a one-line "what it does" note.
_METRIC_TABLE: list[tuple[str, str]] = [
    (
        "Escala de captura (stage 1)",
        "Auditoría del protocolo: garantiza que la distancia, encuadre y luz "
        "son comparables entre baseline y follow-up antes de mirar cualquier "
        "métrica.",
    ),
    (
        "Tamaño de cabeza normalizado (stage 2)",
        "El modelo de fondo (rembg por defecto, GrabCut como respaldo "
        "offline) recorta el fondo y el torso; la cabeza se reescala a "
        "altura fija (800 px) para que cualquier cambio visual sea por "
        "cobertura de pelo, NO por zoom o distancia de cámara.",
    ),
    (
        "Contraste local y detalle (stage 3)",
        "B&W + CLAHE + unsharp mask amplían el contraste de mechones "
        "individuales sin saturar el tono de piel. Permite ver densidad "
        "de cabello fino que en color pasa desapercibido.",
    ),
    (
        "Cluster Lab (stage 4)",
        "K-means en espacio Lab con k=3. Paleta (negro=pelo, coral=piel, "
        "crema=brillos/fondo). El coral es la pista visual directa de "
        "calvicie y zonas de baja densidad.",
    ),
    (
        "% scalp visible (stage 4)",
        "Proporción de píxeles del cluster medio (piel) en el recorte de "
        "cabeza. Un aumento entre A y B indica mayor superficie de scalp "
        "expuesta = thinning. Útil como comparación A vs B, no como "
        "porcentaje clínico absoluto.",
    ),
]


def make_report(
    result: PipelineResult,
    output_path: str | Path,
    title: str = "Athenas — Trichology Report",
    baseline_name: str = "A",
    followup_name: str = "B",
    baseline_date: str | None = None,
    followup_date: str | None = None,
) -> Path:
    """Compose the 4 in-memory figures + a metrics table into a 3-page PDF.

    ``baseline_date`` and ``followup_date`` are optional human-readable
    labels (e.g. ``"2026-01-15"``) that are appended to the per-photo
    photo summary line and the metrics-page header.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Group the 4 stages into 2 pages of 2 figures each.
    panels_per_page: list[list[tuple[str, object]]] = [
        [
            ("1 · Original", result.original),
            ("2 · Sin fondo (cabeza normalizada)", result.no_background),
        ],
        [
            ("3 · Blanco y negro alta definición", result.bw_enhanced),
            ("4 · Cluster Lab (k=3)", result.cluster),
        ],
    ]

    # A4 portrait: 8.5" × 11". Per-panel target height in pixels so the
    # rendered figure fits in roughly half a page once title/header room is
    # accounted for. 110 matches the savefig dpi so conversion is 1:1.
    page_w, page_h = 8.5, 11.0
    panel_target_h = 440  # px, fits comfortably in 5.5" half-page

    # Subtitle line shown on the photo pages — adds the dates between the
    # baseline and follow-up labels when supplied.
    subtitle_a = f"Baseline ({baseline_name})"
    subtitle_b = f"Follow-up ({followup_name})"
    if baseline_date:
        subtitle_a += f"  ·  {baseline_date}"
    if followup_date:
        subtitle_b += f"  ·  {followup_date}"
    photo_summary = f"{subtitle_a}    ×    {subtitle_b}"

    with PdfPages(out) as pdf:
        for page_no, panels in enumerate(panels_per_page, start=1):
            fig, axs = plt.subplots(
                2, 1, figsize=(page_w, page_h),
                gridspec_kw={"hspace": 0.10},
            )
            page_title = title if page_no == 1 else (
                f"{title}  ·  scalp visible: A {result.scalp_percent_a:.1f}%  "
                f"·  B {result.scalp_percent_b:.1f}%"
            )
            fig.suptitle(page_title, fontsize=13, weight="bold", y=0.995)
            # Date subtitle sits just below the title, italic, small.
            fig.text(
                0.5, 0.965,
                photo_summary,
                ha="center", va="top", style="italic", fontsize=10, color="#444",
            )
            for ax, (label, src_fig) in zip(axs, panels, strict=True):
                # Render the source figure to RGB then resize so it fits the
                # half-page slot — the source Figure was sized for
                # target_h=800 which is too tall for half a page.
                rgb = _resize_to_height(fig_to_rgb(src_fig), panel_target_h)
                ax.imshow(rgb)
                # Single title, left-aligned, bold — the only label shown.
                ax.set_title(label, fontsize=11, weight="bold", loc="left")
                ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)
            # Free the source figures once rasterised — no further use.
            for _, src_fig in panels:
                plt.close(src_fig)

        # ---- Page 3: metrics table ------------------------------------
        _write_metrics_page(
            pdf, subtitle_a, subtitle_b,
            scalp_a=result.scalp_percent_a,
            scalp_b=result.scalp_percent_b,
        )

    import logging
    logging.getLogger("athenas").info("PDF report written: %s", out)
    return out


def _resize_to_height(rgb: np.ndarray, target_h: int) -> np.ndarray:
    """Resize ``rgb`` so its height is ``target_h`` px, preserving AR."""
    h, w = rgb.shape[:2]
    if h <= target_h:
        return rgb
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(rgb, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _write_metrics_page(
    pdf: PdfPages,
    baseline_name: str,
    followup_name: str,
    scalp_a: float,
    scalp_b: float,
) -> None:
    """Append a final page that documents every metric in italic prose."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.10, 0.06, 0.82, 0.88])
    ax.axis("off")

    fig.suptitle(
        "Athenas — Métricas y notas",
        fontsize=13, weight="bold", y=0.975,
    )
    # Photo summary line at the top, italic
    ax.text(
        0.0, 1.0,
        f"Baseline ({baseline_name})  ×  Follow-up ({followup_name})  ·  "
        f"scalp visible: A {scalp_a:.1f}%  ·  B {scalp_b:.1f}%",
        ha="left", va="top", transform=ax.transAxes,
        fontsize=10, style="italic", color="#444",
    )

    # Each metric: bold heading + italic body, separated by blank lines.
    cursor_y = 0.93
    line_height = 0.018
    block_gap = 0.035
    text_kwargs = {"ha": "left", "va": "top", "transform": ax.transAxes}

    for name, note in _METRIC_TABLE:
        # Heading line — bold, not italic, identifies the metric
        ax.text(
            0.0, cursor_y, name,
            fontsize=10.5, weight="bold", color="#222",
            **text_kwargs,
        )
        cursor_y -= line_height * 1.2
        # Wrap the note manually at ~95 chars so the layout is predictable.
        for wrapped in _wrap_text(note, width=95):
            ax.text(
                0.0, cursor_y, wrapped,
                fontsize=9.5, color="#444",
                style="italic",
                **text_kwargs,
            )
            cursor_y -= line_height
        cursor_y -= block_gap

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)


def _wrap_text(text: str, width: int) -> list[str]:
    """Greedy wrap onto ``width`` characters, breaking on spaces."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        if not current:
            current = w
        elif len(current) + 1 + len(w) <= width:
            current += " " + w
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def _unused_marker(_x: np.ndarray) -> None:
    """Silence 'np unused' warnings if numpy import is later dropped."""
    return None
