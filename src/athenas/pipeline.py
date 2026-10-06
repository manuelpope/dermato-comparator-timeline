"""Simplified pipeline: build 4 in-memory comparison figures.

MVP2: nothing is written to disk. ``run_pipeline`` returns a ``PipelineResult``
holding four ``matplotlib.figure.Figure`` objects (one per processing stage)
that callers (the report module) can compose into a final PDF.

Stages
------
1. Original (raw frames — useful to detect capture-protocol drift)
2. Background removed (rembg by default, GrabCut as offline fallback) +
   **head-normalised crop** (both A and B scaled to the same display
   height so any zoom/distance difference is normalised away).
3. B&W high-definition (CLAHE + unsharp mask) of the normalised head crop.
4. Lab-space k-means cluster (skin / hair / highlights) on the normalised
   head crop. The "skin" cluster uses coral so exposed scalp POPS visually;
   the figure title shows the % scalp visible so thinning is measured at a
   glance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from .background import (
    apply_mask,
    fill_background_white,
    remove_background_grabcut,
    remove_background_rembg,
)
from .cluster import cluster_lab
from .comparison import head_bbox, head_crop_resize, make_comparison, pad_to_canvas
from .config import PipelineConfig
from .enhance import enhance_bw
from .io import read_rgb

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """In-memory comparison figures + the cluster metric. No paths on disk."""

    original: Figure
    no_background: Figure
    bw_enhanced: Figure
    cluster: Figure
    scalp_percent_a: float = 0.0
    scalp_percent_b: float = 0.0


def _remove_background(rgb: np.ndarray, config: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """Route to the configured background-removal backend.

    Both backends return ``(mask, rgba)`` with the same contract, so the
    downstream pipeline is backend-agnostic.
    """
    method = config.background_method
    if method == "rembg":
        return remove_background_rembg(
            rgb,
            model=config.rembg_model,
            alpha_threshold=config.rembg_alpha_threshold,
        )
    if method == "grabcut":
        return remove_background_grabcut(
            rgb,
            config.grabcut_rect_frac,
            config.grabcut_iterations,
        )
    raise ValueError(
        f"Unknown background_method: {method!r}. Use 'rembg' or 'grabcut'."
    )


def run_pipeline(
    baseline_path: str,
    followup_path: str,
    config: PipelineConfig | None = None,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    date_a: str | None = None,
    date_b: str | None = None,
) -> PipelineResult:
    """Run the 4-stage comparison and return 4 Figure objects in memory.

    Optional overrides:
    - ``label_a`` / ``label_b``: per-photo legend shown on every figure
      (e.g. ``"Baseline (A)"`` → ``"Baseline · 2026-01-15"`` when paired
      with ``date_a``). Defaults to ``"Baseline (A)"`` / ``"Follow-up (B)"``.
    - ``date_a`` / ``date_b``: human-readable date strings appended to
      the legend when supplied.
    """
    if config is None:
        config = PipelineConfig()

    log.info("Reading images")
    a = read_rgb(baseline_path)
    b = read_rgb(followup_path)
    log.info("  A: %s — %s", baseline_path, a.shape)
    log.info("  B: %s — %s", followup_path, b.shape)

    target_h = config.display_height_px

    legend_a = label_a or "Baseline (A)"
    legend_b = label_b or "Follow-up (B)"
    if date_a:
        legend_a = f"{legend_a}  ·  {date_a}"
    if date_b:
        legend_b = f"{legend_b}  ·  {date_b}"

    # ---- Stage 1: original (no normalisation — capture-protocol audit) ---
    log.info("Stage 1: original")
    f1 = make_comparison(
        a, b,
        title="1 · Original",
        label_a=legend_a, label_b=legend_b,
        target_h=target_h,
        with_suptitles=False,
    )

    # ---- Background removal + head bbox ---------------------------------
    log.info("Background removal (%s)", config.background_method)
    mask_a, _ = _remove_background(a, config)
    mask_b, _ = _remove_background(b, config)
    no_bg_a = apply_mask(a, mask_a)
    no_bg_b = apply_mask(b, mask_b)

    bbox_a = head_bbox(
        mask_a,
        padding_frac=config.head_padding_frac,
        cutoff_ratio=config.head_cutoff_ratio,
    )
    bbox_b = head_bbox(
        mask_b,
        padding_frac=config.head_padding_frac,
        cutoff_ratio=config.head_cutoff_ratio,
    )
    log.info("  Head bbox A: %s — B: %s", bbox_a, bbox_b)

    # Crop both heads to the same display height — this is the size
    # normalisation: from this point on, the two heads occupy the same
    # vertical extent in any figure, so any visual difference is due to
    # hair coverage / colour, NOT zoom.
    head_a = head_crop_resize(no_bg_a, bbox_a, target_h=target_h)
    head_b = head_crop_resize(no_bg_b, bbox_b, target_h=target_h)

    # For the cluster we want the *original* head RGB colours, just with
    # the background filled white (not zeroed) so the k-means still picks
    # "hair" as the darkest cluster. Without this fill, the BLACK
    # background would steal the "hair" slot and the metric would be wrong.
    white_a = head_crop_resize(
        fill_background_white(a, mask_a), bbox_a, target_h=target_h,
    )
    white_b = head_crop_resize(
        fill_background_white(b, mask_b), bbox_b, target_h=target_h,
    )

    # Force BOTH heads (and the cluster inputs) into the same canvas so the
    # side-by-side panels have identical size — height is already matched
    # by ``head_crop_resize``; equalising the width by padding the smaller
    # one with white guarantees a fair visual comparison regardless of
    # capture distance. The padding lands in the "highlight" cluster so
    # the % scalp visible still tracks hair coverage cleanly.
    target_w = max(head_a.shape[1], head_b.shape[1],
                    white_a.shape[1], white_b.shape[1])
    target_shape = (target_h, target_w)
    head_a = pad_to_canvas(head_a, target_shape)
    head_b = pad_to_canvas(head_b, target_shape)
    white_a = pad_to_canvas(white_a, target_shape)
    white_b = pad_to_canvas(white_b, target_shape)

    # ---- Stage 2: background removed + normalised ----------------------
    log.info("Stage 2: sin fondo + head-normalised")
    f2 = make_comparison(
        head_a, head_b,
        title="2 · Sin fondo (cabeza normalizada)",
        label_a=legend_a.replace("Baseline", "A").replace("Follow-up", "B"),
        label_b=legend_b.replace("Baseline", "A").replace("Follow-up", "B"),
        target_h=target_h,
        with_suptitles=False,
    )

    # ---- Stage 3: B&W high definition on normalised head ---------------
    log.info("Stage 3: B&W high-definition on normalised head")
    bw_a = enhance_bw(
        head_a, clip_limit=config.clahe_clip_limit, grid=config.clahe_grid,
        amount=config.unsharp_amount, radius=config.unsharp_radius,
    )
    bw_b = enhance_bw(
        head_b, clip_limit=config.clahe_clip_limit, grid=config.clahe_grid,
        amount=config.unsharp_amount, radius=config.unsharp_radius,
    )
    f3 = make_comparison(
        bw_a, bw_b,
        title="3 · Blanco y negro alta definición",
        label_a="A — B&W HD", label_b="B — B&W HD",
        cmap="gray", target_h=target_h,
        with_suptitles=False,
    )

    # ---- Stage 4: Lab cluster on normalised head -----------------------
    log.info("Stage 4: Lab-space k-means cluster (k=%d) on normalised head", config.cluster_k)
    res_a = cluster_lab(white_a, k=config.cluster_k, seed=config.cluster_seed)
    res_b = cluster_lab(white_b, k=config.cluster_k, seed=config.cluster_seed)

    # Index 1 = mid-L = "skin / scalp" for k=3. With k>3 the user must rely
    # on the colour cue, so we cap the displayed metric at the mid cluster.
    scalp_idx = 1 if config.cluster_k >= 3 else 0
    pct_a = res_a.proportions[scalp_idx] * 100
    pct_b = res_b.proportions[scalp_idx] * 100

    title4 = (
        f"4 · Cluster Lab (k={config.cluster_k}) — "
        f"scalp visible: A {pct_a:4.1f}%  ·  B {pct_b:4.1f}%"
    )
    f4 = make_comparison(
        res_a.rgb, res_b.rgb,
        title=title4,
        label_a=f"A — cluster  ({pct_a:.1f}% scalp)",
        label_b=f"B — cluster  ({pct_b:.1f}% scalp)",
        target_h=target_h,
        with_suptitles=False,
    )

    log.info("Done. 4 in-memory figures ready.")
    return PipelineResult(
        original=f1,
        no_background=f2,
        bw_enhanced=f3,
        cluster=f4,
        scalp_percent_a=pct_a,
        scalp_percent_b=pct_b,
    )
