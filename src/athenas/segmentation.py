"""Light hair-structure segmentation on aligned ROIs.

The mask represents *visual structures compatible with hair* under the
photographic protocol — it is **not** a follicle detector.
"""

from __future__ import annotations

import cv2
import numpy as np


def hair_structure_mask(
    rgb: np.ndarray,
    head_mask: np.ndarray | None = None,
    dark_percentile: int = 35,
    gradient_percentile: int = 58,
    morph_kernel: int = 3,
) -> tuple[np.ndarray, dict[str, float]]:
    """Combine local darkness and gradient magnitude to detect hair-like pixels.

    The thresholds are computed from the percentiles of the *valid* pixels
    (those inside ``head_mask`` if provided), which makes the result robust
    against backgrounds with very different intensities.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grayf = gray.astype(np.float32)

    # Local-mean division: dark relative to neighbours.
    local = cv2.GaussianBlur(grayf, (0, 0), 13)
    rel = grayf / (local + 1e-5)

    # Gradient magnitude: hair has strong local edges.
    gx = cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    valid = head_mask > 0 if head_mask is not None else np.ones(gray.shape, dtype=bool)

    t_dark = float(np.percentile(rel[valid], dark_percentile))
    t_grad = float(np.percentile(grad[valid], gradient_percentile))

    m1 = rel < t_dark
    m2 = grad > t_grad
    mask = ((m1 & m2) & valid).astype(np.uint8) * 255

    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask, {"t_dark": t_dark, "t_grad": t_grad}
