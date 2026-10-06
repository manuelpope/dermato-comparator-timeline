"""Default configuration for the simplified Athenas comparison pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline configuration."""

    # --- Background removal ----------------------------------------------
    # Selects the algorithm. ``rembg`` is a deep-learning matting model
    # that preserves hair edges much better than GrabCut but downloads
    # ~200 MB on first run. ``grabcut`` is the OpenCV GMM-based fallback
    # that ships with the project and is fully offline.
    background_method: str = "grabcut"  # one of: "rembg", "grabcut"
    # rembg-specific knobs (used when background_method == "rembg")
    rembg_model: str = "birefnet-portrait"
    rembg_alpha_threshold: int = 32  # pixels with alpha < this become bg
    # GrabCut-specific knobs (used when background_method == "grabcut")
    grabcut_rect_frac: tuple[float, float, float, float] = (0.05, 0.03, 0.90, 0.94)
    grabcut_iterations: int = 5

    # --- Hair-structure segmentation (kept available, not used by default) -
    dark_percentile: int = 35
    gradient_percentile: int = 58
    morph_kernel: int = 3

    # --- B&W enhancement --------------------------------------------------
    clahe_clip_limit: float = 3.0
    clahe_grid: int = 8
    unsharp_amount: float = 1.5
    unsharp_radius: int = 3

    # --- Lab cluster (stage 4) --------------------------------------------
    cluster_k: int = 3
    cluster_seed: int = 42

    # --- Comparison layout ------------------------------------------------
    display_height_px: int = 800
    # Head-cropping: the bbox gets a generous border of black void around
    # the head silhouette (visual breathing room) and the bottom is
    # capped well above the torso so the buzo doesn't pollute the
    # cluster metric.
    head_padding_frac: float = 0.45  # padding around the head mask bbox when cropping
    head_cutoff_ratio: float = 0.65  # fraction of bbox height treated as "head"

    # --- Output -----------------------------------------------------------
    output_dir: Path = Path("outputs")
    dpi: int = 150
