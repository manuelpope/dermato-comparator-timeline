"""B&W enhancement: grayscale + CLAHE + unsharp mask.

The combination boosts local contrast (CLAHE) and edges (unsharp), which makes
hair-strand patterns easier to inspect than the raw color photograph.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to a single-channel ``uint8`` grayscale image."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def clahe_enhance(
    gray: np.ndarray,
    clip_limit: float = 3.0,
    grid: int = 8,
) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalisation."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def unsharp_mask(
    gray: np.ndarray,
    amount: float = 1.5,
    radius: int = 3,
) -> np.ndarray:
    """Sharpen via unsharp mask: ``out = gray + amount * (gray - blur)``."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=radius)
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_bw(
    rgb: np.ndarray,
    clip_limit: float = 3.0,
    grid: int = 8,
    amount: float = 1.5,
    radius: int = 3,
) -> np.ndarray:
    """Return a B&W, contrast-boosted, sharpened ``uint8`` grayscale image."""
    gray = to_grayscale(rgb)
    gray = clahe_enhance(gray, clip_limit=clip_limit, grid=grid)
    gray = unsharp_mask(gray, amount=amount, radius=radius)
    return gray
