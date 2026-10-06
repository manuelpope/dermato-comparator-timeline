"""Four visual filters for skin-lesion comparison.

Each filter takes an RGB uint8 ndarray (H×W×3) and returns either an
RGB uint8 or a gray uint8 ndarray (the PDF renderer expands gray to
RGB before plotting).

The recipes are intentionally thin wrappers — they reuse
:func:`athenas.enhance.clahe_enhance` and
:func:`athenas.enhance.enhance_bw` where possible to keep the trichology
and dermatology pipelines consistent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import cv2
import numpy as np

from athenas.comparison import fit_to_canvas
from athenas.enhance import clahe_enhance, enhance_bw

# ---------------------------------------------------------------------------
# Single-image filters (synchronous, called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def filter_clahe(rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Color-preserving CLAHE on the Lab-L channel.

    Boosts local luminance contrast without disturbing the a*/b* colour
    axes. Default ``clip_limit=2.0`` is more conservative than the
    trichology default of 3.0 to avoid burning subtle erythema.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    L = clahe_enhance(L, clip_limit=clip_limit)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2RGB)


def filter_bw_contrast(rgb: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE + unsharp mask. Returns 2-D ``uint8``.

    Delegates straight to :func:`athenas.enhance.enhance_bw` with the
    trichology defaults — they work fine on small colour crops too.
    """
    return enhance_bw(rgb)


def filter_color_contrast(
    rgb: np.ndarray,
    clip_limit: float = 2.0,
    saturation_boost: float = 1.2,
) -> np.ndarray:
    """Luminance contrast (Lab-L CLAHE) + chroma stretch (HSV-S × boost).

    The two-axis boost is what makes this filter visually distinct from
    :func:`filter_clahe`: the Lab-L pass brightens dark areas, the
    HSV-S pass pushes colours apart. ``saturation_boost`` is capped at
    ``1.5`` in practice to avoid clipping erythema/cyanosis cues.
    """
    # 1. Lab-L CLAHE — preserves hue, only brightens.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    L = clahe_enhance(L, clip_limit=clip_limit)
    rgb_lab = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2RGB)

    # 2. HSV-S stretch — pushes colours apart.
    hsv = cv2.cvtColor(rgb_lab, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_boost, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def filter_scale(rgb: np.ndarray, target_h: int = 600) -> np.ndarray:
    """Resize to a fixed display height so temporal shots compare 1:1."""
    return fit_to_canvas(rgb, target_h=target_h)


# ---------------------------------------------------------------------------
# Parallel orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LesionFilters:
    """The 4 filter outputs for one lesion image."""

    clahe: np.ndarray  # RGB uint8
    bw: np.ndarray  # gray uint8 (2-D)
    color: np.ndarray  # RGB uint8
    normalised: np.ndarray  # RGB uint8 (resized to target_h)


@dataclass(frozen=True)
class LesionResult:
    """One lesion's display name + its 4 filter outputs."""

    name: str
    filters: LesionFilters


async def parallel_apply_filters(
    image: np.ndarray,
    *,
    target_height: int,
    clahe_clip_limit: float,
    saturation_boost: float,
) -> LesionFilters:
    """Apply the 4 filters to ``image`` in parallel via asyncio.to_thread.

    Each filter call is short (sub-second for typical clinical photos), so
    the parallel approach is bounded by the slowest filter (usually Lab
    conversions). All four honour `nyi` and run on the GIL-aware stdlib pool.
    """
    clahe, bw, color, scaled = await asyncio.gather(
        asyncio.to_thread(filter_clahe, image, clahe_clip_limit),
        asyncio.to_thread(filter_bw_contrast, image),
        asyncio.to_thread(
            filter_color_contrast, image, clahe_clip_limit, saturation_boost,
        ),
        asyncio.to_thread(filter_scale, image, target_height),
    )
    return LesionFilters(clahe=clahe, bw=bw, color=color, normalised=scaled)
