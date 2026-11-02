"""Morphological wrinkle-line detection per facial region.

The MVP algorithm is intentionally simple and parameterised so the user can
tune it visually:

1. Convert the face to grayscale.
2. For each facial region (forehead, glabella, periorbital L/R, nasolabial
   L/R, perioral) build a convex-hull ROI from the region's landmarks.
3. Inside the ROI, run a morphological black-hat using two anisotropic
   kernels (horizontal and vertical) to capture line-like structures in
   any orientation.
4. Threshold by the 92-nd percentile inside the ROI (adaptive, tolerates
   exposure changes between photos).
5. Close the binary mask with a small elliptical kernel to merge pixels
   belonging to the same line.
6. Filter connected components to keep only those with at least one
   dimension ≥ ``min_line_length_px``.

Public dataclasses
------------------
- :class:`WrinkleRegionResult`
- :class:`WrinkleDetectionResult`

Public functions
----------------
- :func:`detect_wrinkles_in_region` — single-region black-hat pipeline.
- :func:`build_region_mask` — convex hull of landmark indices → ROI mask.
- :func:`run_wrinkle_detection` — apply the pipeline to every region in
  ``WRINKLE_REGIONS``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from athenas.face_expression.face_detection import (
    FOREHEAD_REGION_INDICES,
    GLABELLA_REGION_INDICES,
    NASOLABIAL_L_INDICES,
    NASOLABIAL_R_INDICES,
    PERIORAL_REGION_INDICES,
    PERIORBITAL_L_INDICES,
    PERIORBITAL_R_INDICES,
    FaceLandmarks,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WrinkleRegionResult:
    """Wrinkle-line detection result for one facial region.

    Attributes
    ----------
    region
        Region name (key of ``WRINKLE_REGIONS``).
    indices
        Landmark indices that define the region.
    density
        Proportion of pixels inside the region mask classified as wrinkle
        pixels (in ``[0, 1]``).
    mask
        ``(H, W)`` ``bool`` ndarray — wrinkle pixels in this region.
    annotated_overlay
        ``(H, W, 3)`` ``uint8`` ndarray — magenta overlay on the original
        image showing the detected wrinkles.
    """

    region: str
    indices: tuple[int, ...]
    density: float
    mask: np.ndarray
    annotated_overlay: np.ndarray


@dataclass(frozen=True)
class WrinkleDetectionResult:
    """Wrinkle-line detection across every configured region."""

    per_region: dict[str, WrinkleRegionResult]
    combined_mask: np.ndarray      # (H, W) bool — OR of every region's mask
    annotated_face: np.ndarray     # (H, W, 3) uint8 — face with all lines drawn


# ---------------------------------------------------------------------------
# Region definitions — one entry per wrinkle region we want to score.
# ---------------------------------------------------------------------------

WRINKLE_REGIONS: dict[str, tuple[int, ...]] = {
    "frontal":       FOREHEAD_REGION_INDICES,
    "glabellar":     GLABELLA_REGION_INDICES,
    "periorbital_L": PERIORBITAL_L_INDICES,
    "periorbital_R": PERIORBITAL_R_INDICES,
    "nasolabial_L":  NASOLABIAL_L_INDICES,
    "nasolabial_R":  NASOLABIAL_R_INDICES,
    "perioral":      PERIORAL_REGION_INDICES,
}


# ---------------------------------------------------------------------------
# Region mask
# ---------------------------------------------------------------------------


def build_region_mask(
    landmarks_xy: np.ndarray,
    indices: tuple[int, ...],
    image_shape_hw: tuple[int, int],
    *,
    fill_convex_hull: bool = True,
) -> np.ndarray:
    """Convex hull of the region's landmarks → ``(H, W)`` ``bool`` mask.

    ``fill_convex_hull=True`` fills the polygon; ``False`` only outlines it.
    We always fill — the wrinkle detector needs pixels inside the region,
    not just on its boundary.
    """
    h, w = image_shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    # Filter landmarks that fall inside the image bounds.
    pts = landmarks_xy[list(indices)]
    in_bounds = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = pts[in_bounds]
    if len(pts) < 3:
        return mask.astype(bool)

    pts_int = pts.astype(np.int32).reshape(-1, 1, 2)
    if fill_convex_hull:
        hull = cv2.convexHull(pts_int)
        cv2.fillConvexPoly(mask, hull, 1)
    else:
        cv2.polylines(mask, [pts_int], isClosed=True, color=1, thickness=1)

    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Single-region wrinkle detection
# ---------------------------------------------------------------------------


def _multi_orient_blackhat(
    gray: np.ndarray,
    kernel_length: int,
    orientations_deg: tuple[int, ...] = (0, 30, 60, 90, 120, 150),
) -> np.ndarray:
    """Black-hat with a thin line kernel rotated to ``orientations_deg``.

    Real wrinkles are rarely perfectly horizontal/vertical — a 30° or 60°
    glabellar line is invisible to ``cv2.MORPH_RECT (k, 1)``. We therefore
    build an oriented thin-ellipse kernel via warpAffine and take the
    per-pixel max over all rotations. Returns ``uint8``.
    """
    k = max(3, kernel_length)
    # Build a thin horizontal ellipse: height 1 px, width k px.
    base = np.zeros((k, k), dtype=np.uint8)
    cv2.ellipse(base, (k // 2, k // 2), (k // 2, 0), 0, 0, 360, 1, thickness=-1)

    response = np.zeros(gray.shape, dtype=np.uint8)
    for angle in orientations_deg:
        M = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1.0)
        kern = (cv2.warpAffine(base, M, (k, k), borderValue=0) > 0).astype(np.uint8)
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kern)
        response = cv2.max(response, bh)
    return response


def _hessian_ridge_response(gray: np.ndarray) -> np.ndarray:
    """Ridge / valley "strength" from the Hessian of the grayscale image.

    For a line-like structure one eigenvalue of the Hessian has a large
    absolute value (perpendicular to the line) while the other is near zero
    (along the line). The smaller-magnitude eigenvalue is therefore a
    good ridge-ness indicator. Returns a ``uint8`` image normalised to
    ``[0, 255]``.
    """
    g32 = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(g32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g32, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3)
    gxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1, ksize=3)
    gyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)
    trace = gxx + gyy
    det = gxx * gyy - gxy * gxy
    disc = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0))
    l1 = (trace + disc) / 2
    l2 = (trace - disc) / 2
    ridgeness = np.minimum(np.abs(l1), np.abs(l2))
    m = float(ridgeness.max())
    if m <= 0:
        return np.zeros_like(gray)
    return np.clip((ridgeness / m) * 255.0, 0, 255).astype(np.uint8)


def detect_wrinkles_in_region(
    gray: np.ndarray,
    region_mask: np.ndarray,
    *,
    kernel_length: int = 15,
    min_line_length_px: int = 10,
    closing_kernel: int = 3,
    threshold_percentile: float = 80.0,
    orientations_deg: tuple[int, ...] = (0, 30, 60, 90, 120, 150),
    use_hessian: bool = True,
) -> np.ndarray:
    """Detect wrinkle pixels inside ``region_mask`` on a grayscale image.

    Algorithm
    ---------
    1. ``multi-orient black-hat`` — a thin ellipse kernel applied at
       ``orientations_deg`` rotations (default 6 angles), per-pixel
       maximum. This catches diagonal wrinkles that ``cv2.MORPH_RECT``
       misses.
    2. ``Hessian ridge response`` — the smaller-magnitude eigenvalue of
       the image Hessian, which fires on thin line-like structures
       (ridges/valleys). Combined with the black-hat via per-pixel max
       (configurable via ``use_hessian``).
    3. Adaptive percentile threshold inside the ROI.
    4. Morphological closing (configurable kernel).
    5. Connected-component filter — keep only components whose width or
       height is ≥ ``min_line_length_px``.

    Parameters
    ----------
    gray
        ``(H, W)`` ``uint8`` grayscale image.
    region_mask
        ``(H, W)`` ``bool`` ROI mask.
    kernel_length
        Length (in pixels) of the oriented ellipse kernel.
    min_line_length_px
        Connected components shorter than this are dropped as noise.
        Lowered to 10 (from 12) to catch finer lines.
    closing_kernel
        Radius of the morphological closing. ``0`` disables it.
    threshold_percentile
        Percentile used for the binary threshold (within the ROI). 80 is a
        good starting point; the previous default of 92 was too strict on
        real photos where wrinkles contribute <10 % of the response.
    orientations_deg
        Rotations (degrees) of the kernel. The default covers every 30°.
    use_hessian
        If ``True`` (default), the Hessian ridge response is combined
        with the black-hat response via per-pixel max — strong wrinkles
        fire on both signals.

    Returns
    -------
    np.ndarray
        ``(H, W)`` ``bool`` mask of detected wrinkle pixels.
    """
    if gray.ndim != 2:
        raise ValueError(f"gray must be (H, W), got {gray.shape}")
    if region_mask.shape != gray.shape:
        raise ValueError("region_mask and gray must have the same shape")
    if not region_mask.any():
        return np.zeros_like(gray, dtype=bool)

    # 1 + 2 — combined response.
    bh = _multi_orient_blackhat(gray, kernel_length, orientations_deg).astype(np.float32)
    if use_hessian:
        hess = _hessian_ridge_response(gray).astype(np.float32)
        response = np.maximum(bh, hess)
    else:
        response = bh

    # 3 — adaptive threshold inside the ROI. Tiny regions (glabella has
    # only ~few hundred px on a real photo) need a lower percentile, otherwise
    # the "above" set is too sparse to drive any connected component over
    # ``min_line_length_px``. Scale the percentile with the region size.
    roi_pixels = response[region_mask]
    roi_count = int(roi_pixels.size)
    if roi_count < 800:
        # Very small region → trust the response's shape, not the percentile.
        effective_percentile = max(50.0, threshold_percentile - 25.0)
    elif roi_count < 3000:
        effective_percentile = max(60.0, threshold_percentile - 12.0)
    else:
        effective_percentile = threshold_percentile
    threshold_value = float(np.percentile(roi_pixels, effective_percentile))
    candidate = (response > threshold_value) & region_mask

    # 4 — closing.
    if closing_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel)
        )
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)

    # 5 — connected-component filter (line-length).
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8,
    )
    filtered = np.zeros_like(candidate, dtype=bool)
    for i in range(1, nlabels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w >= min_line_length_px or h >= min_line_length_px:
            filtered[labels == i] = True

    return filtered


# ---------------------------------------------------------------------------
# Per-region orchestrator
# ---------------------------------------------------------------------------


_OVERLAY_TINT_BGR = (255, 0, 255)  # magenta


def _overlay_wrinkles_on_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    """Paint a magenta alpha-blend where ``mask`` is True."""
    if rgb.shape[:2] != mask.shape:
        raise ValueError("rgb and mask shapes don't match")
    out = rgb.astype(np.float32).copy()
    tint = np.zeros_like(out)
    tint[..., 0] = _OVERLAY_TINT_BGR[0]
    tint[..., 1] = _OVERLAY_TINT_BGR[1]
    tint[..., 2] = _OVERLAY_TINT_BGR[2]
    blend_mask = mask & (mask.sum(axis=-1, keepdims=True) > 0)
    out[blend_mask] = (1 - alpha) * out[blend_mask] + alpha * tint[blend_mask]
    return out.astype(np.uint8)


def run_wrinkle_detection(
    rgb: np.ndarray,
    landmarks: FaceLandmarks,
    *,
    region_definitions: dict[str, tuple[int, ...]] | None = None,
    kernel_length: int = 15,
    min_line_length_px: int = 12,
    closing_kernel: int = 3,
    threshold_percentile: float = 92.0,
) -> WrinkleDetectionResult:
    """Run wrinkle detection on every region of ``landmarks``.

    Parameters
    ----------
    rgb
        ``(H, W, 3)`` ``uint8`` RGB image (used to build the overlay).
    landmarks
        Face landmarks from :func:`face_detection.detect_landmarks`.
    region_definitions
        Override for ``WRINKLE_REGIONS`` (e.g. to test a single region).
    kernel_length, min_line_length_px, closing_kernel, threshold_percentile
        Forwarded to :func:`detect_wrinkles_in_region`.

    Returns
    -------
    WrinkleDetectionResult
    """
    regions = region_definitions or WRINKLE_REGIONS
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    per_region: dict[str, WrinkleRegionResult] = {}
    combined = np.zeros((h, w), dtype=bool)
    annotated = rgb.copy()

    for name, indices in regions.items():
        region_mask = build_region_mask(landmarks.pixels_xy, indices, (h, w))
        if not region_mask.any():
            per_region[name] = WrinkleRegionResult(
                region=name,
                indices=indices,
                density=0.0,
                mask=combined,  # empty mask — placeholder
                annotated_overlay=annotated,
            )
            continue

        wrinkle_mask = detect_wrinkles_in_region(
            gray,
            region_mask,
            kernel_length=kernel_length,
            min_line_length_px=min_line_length_px,
            closing_kernel=closing_kernel,
            threshold_percentile=threshold_percentile,
        )
        density = float(wrinkle_mask.sum() / max(region_mask.sum(), 1))
        overlay = _overlay_wrinkles_on_rgb(annotated, wrinkle_mask)

        per_region[name] = WrinkleRegionResult(
            region=name,
            indices=indices,
            density=density,
            mask=wrinkle_mask,
            annotated_overlay=overlay,
        )
        combined |= wrinkle_mask
        # Update the running overlay so subsequent regions compose.
        annotated = _overlay_wrinkles_on_rgb(annotated, wrinkle_mask)

    return WrinkleDetectionResult(
        per_region=per_region,
        combined_mask=combined,
        annotated_face=annotated,
    )


# ---------------------------------------------------------------------------
# Expression-induced mask (wrinkles that APPEAR with the expression).
# ---------------------------------------------------------------------------


def compute_expression_induced_mask(
    wrinkles_relaxed: WrinkleDetectionResult,
    wrinkles_expression: WrinkleDetectionResult,
) -> np.ndarray:
    """Pixels classified as wrinkles ONLY in the expression photo.

    This is the set-difference ``mask_expression AND NOT mask_relaxed`` —
    the new lines that the relaxed skin was hiding until you moved the face.
    Returned as a ``(H, W)`` ``bool`` mask matching the photo canvas.

    If the two masks come from photos with different shapes, the smaller
    one is zero-padded to the larger before the set difference is taken.
    """
    mask_r = wrinkles_relaxed.combined_mask
    mask_e = wrinkles_expression.combined_mask

    if mask_r.shape != mask_e.shape:
        # Pad the smaller one to the union shape (right/bottom aligned).
        h = max(mask_r.shape[0], mask_e.shape[0])
        w = max(mask_r.shape[1], mask_e.shape[1])
        padded_r = np.zeros((h, w), dtype=bool)
        padded_e = np.zeros((h, w), dtype=bool)
        padded_r[: mask_r.shape[0], : mask_r.shape[1]] = mask_r
        padded_e[: mask_e.shape[0], : mask_e.shape[1]] = mask_e
        mask_r, mask_e = padded_r, padded_e

    return mask_e & ~mask_r
