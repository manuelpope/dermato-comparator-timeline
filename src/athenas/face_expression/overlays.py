"""Visual overlays used by the aesthetics (Botox before/after) report.

The pipeline produces four views of the *same* normalised face:

===============  =====================================================
View             What the clinician sees
===============  =====================================================
``normalised``   The aligned, cropped, head-size-normalised photo.
``wrinkles``     The same photo with every detected expression line
                 painted in magenta.
``bw``           Grayscale + CLAHE + unsharp — reveals the relief of
                 the skin (shallow grooves read as dark ridges).
``change``       INFERNO heatmap of ``|before − after|`` on the aligned
                 faces — shows *where* the botox changed something.
===============  =====================================================

Everything here is pure NumPy/OpenCV — no matplotlib, no disk I/O — so the
overlays can be built in a worker thread and handed to the PDF renderer.
"""

from __future__ import annotations

import cv2
import numpy as np

from athenas.face_expression.face_detection import (
    FACE_OVAL_INDICES,
    FaceLandmarks,
)
from athenas.face_expression.wrinkle_detection import WrinkleDetectionResult

#: Magenta tint used to paint the detected expression lines (RGB).
_WRINKLE_TINT_RGB = (255, 0, 255)

#: Blend strength of the change heatmap over the "after" photo.
_CHANGE_ALPHA = 0.7


def render_landmarks_overlay(
    rgb: np.ndarray,
    landmarks: FaceLandmarks,
    *,
    face_oval_color: tuple[int, int, int] = (0, 255, 0),
    point_color: tuple[int, int, int] = (0, 255, 0),
    point_radius: int = 1,
) -> np.ndarray:
    """Draw the face oval + every landmark on a copy of ``rgb``."""
    out = rgb.copy()
    h, w = rgb.shape[:2]

    oval_pts = landmarks.pixels_xy[list(FACE_OVAL_INDICES)].astype(np.int32)
    cv2.polylines(
        out,
        [oval_pts.reshape(-1, 1, 2)],
        isClosed=True,
        color=face_oval_color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    pts = landmarks.pixels_xy.astype(np.int32)
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(
                out, (int(x), int(y)), point_radius, point_color, -1, cv2.LINE_AA,
            )
    return out


def render_wrinkles_overlay(
    rgb: np.ndarray,
    wrinkles: WrinkleDetectionResult,
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    """Paint every detected wrinkle region in magenta on a copy of ``rgb``."""
    h, w = rgb.shape[:2]
    for region in wrinkles.per_region.values():
        if region.mask.shape != (h, w):
            raise ValueError(
                f"wrinkle region '{region.region}' mask shape "
                f"{region.mask.shape} doesn't match rgb shape {(h, w)}"
            )

    out = rgb.astype(np.float32).copy()
    tint = np.array(_WRINKLE_TINT_RGB, dtype=np.float32)
    for region in wrinkles.per_region.values():
        mask = region.mask
        if not mask.any():
            continue
        out[mask] = (1 - alpha) * out[mask] + alpha * tint
    return out.astype(np.uint8)


def render_exaggerated_lines(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    ink_color: tuple[int, int, int] = (40, 20, 20),
    dot_radius_px: int = 2,
    dot_alpha: float = 0.95,
) -> np.ndarray:
    """Stamp every pixel of ``mask`` as a small dark dot over ``rgb``.

    This is the "caricature" view: the lines the detector found are
    painted as discrete dots, which reads as *these are the lines we can
    work on* without the posterised look a heavy blur would give.
    """
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"rgb shape {rgb.shape[:2]} != mask shape {mask.shape}")

    out = rgb.astype(np.float32).copy()
    if not mask.any():
        return out.astype(np.uint8)

    dilated = mask
    if dot_radius_px >= 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dot_radius_px * 2 + 1,) * 2,
        )
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    tint = np.array(ink_color, dtype=np.float32)
    out[dilated] = (1 - dot_alpha) * out[dilated] + dot_alpha * tint
    return out.astype(np.uint8)


def render_change_heatmap(
    rgb_before: np.ndarray,
    rgb_after: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    alpha: float = _CHANGE_ALPHA,
) -> np.ndarray:
    """INFERNO heatmap of the absolute difference between two aligned faces.

    The "after" photo is the background (so the clinician keeps the
    anatomical context) and the difference is painted on top.

    ``mask`` (usually the face oval) restricts both the normalisation and
    the painting to the face — otherwise a different background or a
    different shirt would dominate the colour range and hide the real
    change.
    """
    if rgb_before.shape != rgb_after.shape:
        raise ValueError(
            f"before {rgb_before.shape[:2]} and after {rgb_after.shape[:2]} "
            "must have the same size to be compared"
        )

    gray_before = cv2.cvtColor(rgb_before, cv2.COLOR_RGB2GRAY)
    gray_after = cv2.cvtColor(rgb_after, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray_before, gray_after)

    if mask is not None:
        if mask.shape != diff.shape:
            raise ValueError(
                f"mask shape {mask.shape} doesn't match image shape {diff.shape}"
            )
        inside = diff[mask]
        peak = float(inside.max()) if inside.size else 0.0
        if peak > 0:
            diff = np.clip(diff.astype(np.float32) * (255.0 / peak), 0, 255)
        diff = diff.astype(np.uint8)
        heat_bgr = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        out = rgb_after.astype(np.float32).copy()
        out[mask] = (
            (1 - alpha) * out[mask] + alpha * heat_rgb[mask].astype(np.float32)
        )
    else:
        peak = float(diff.max())
        if peak > 0:
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        heat_bgr = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        out = rgb_after.astype(np.float32).copy()
        out = (1 - alpha) * out + alpha * heat_rgb.astype(np.float32)

    return out.clip(0, 255).astype(np.uint8)


def concat_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    *,
    gap_px: int = 12,
    background: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Put two images next to each other, same height, with a white gap.

    Both inputs must already be cropped/resized to a common height — the
    caller is responsible for the normalisation; this helper only glues.
    """
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"left height {left.shape[0]} != right height {right.shape[0]}"
        )
    if left.shape[2] != right.shape[2]:
        raise ValueError("left and right must have the same number of channels")

    h = left.shape[0]
    gap = np.full((h, gap_px, left.shape[2]), background, dtype=left.dtype)
    return np.concatenate([left, gap, right], axis=1)
