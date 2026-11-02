"""Visual treatments of a normalised face.

Every filter receives an RGB ``uint8`` ndarray that has **already** been
aligned, cropped and resized to a common head size — that normalisation is
what makes a before/after pair comparable at all (see
:mod:`app.services.aesthetics.analysis`).

Filters
-------
- ``normalised`` — the reference view (colour, aligned).
- ``wrinkles``   — the detected expression lines painted in magenta.
- ``bw``         — grayscale + CLAHE + unsharp; the *relief* of the skin.
- ``change``     — INFERNO heatmap of the before→after difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from athenas.enhance import enhance_bw
from athenas.face_expression.overlays import (
    render_change_heatmap,
    render_wrinkles_overlay,
)
from athenas.face_expression.wrinkle_detection import WrinkleDetectionResult


@dataclass(frozen=True)
class FaceView:
    """The three visual treatments of one normalised face."""

    #: RGB — the aligned, cropped, head-size-normalised photo.
    normalised: np.ndarray
    #: RGB — the same photo with the detected expression lines in magenta.
    wrinkles: np.ndarray
    #: 2-D gray — CLAHE + unsharp, makes the skin relief readable.
    bw: np.ndarray


@dataclass(frozen=True)
class AestheticsViews:
    """Everything the PDF needs to draw the before/after comparison."""

    before: FaceView
    after: FaceView
    #: RGB — INFERNO heatmap of |before − after| on the aligned faces.
    change_map: np.ndarray


def build_face_view(
    rgb: np.ndarray,
    wrinkles: WrinkleDetectionResult,
) -> FaceView:
    """Build the three views of a single normalised face."""
    return FaceView(
        normalised=rgb,
        wrinkles=render_wrinkles_overlay(rgb, wrinkles),
        bw=enhance_bw(rgb),
    )


def build_change_map(
    rgb_before: np.ndarray,
    rgb_after: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build the INFERNO difference heatmap of two aligned faces.

    ``mask`` (the face oval) keeps the heatmap focused on the face.
    """
    return render_change_heatmap(rgb_before, rgb_after, mask=mask)
