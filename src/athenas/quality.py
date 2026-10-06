"""Image-quality metrics: exposure, contrast, focus."""

from __future__ import annotations

import cv2
import numpy as np


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance, float in ``[0, 1]``."""
    rgbf = rgb.astype(np.float32) / 255.0
    return 0.2126 * rgbf[..., 0] + 0.7152 * rgbf[..., 1] + 0.0722 * rgbf[..., 2]


def quality_metrics(rgb: np.ndarray) -> dict[str, float]:
    """Return a dict of global image-quality descriptors.

    * ``mean_gray`` / ``std_gray`` — exposure and contrast.
    * ``laplacian_variance`` — focus/sharpness proxy.
    * ``dark_clip_pct`` / ``bright_clip_pct`` — proportion of pixels at 0/255.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    hist = np.bincount(gray.ravel(), minlength=256) / gray.size
    return {
        "mean_gray": float(gray.mean()),
        "std_gray": float(gray.std()),
        "laplacian_variance": float(lap.var()),
        "dark_clip_pct": float(hist[:3].sum() * 100),
        "bright_clip_pct": float(hist[-3:].sum() * 100),
    }


def focus_map(rgb: np.ndarray, grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Return a ``rows × cols`` array with the Laplacian variance of each cell.

    Used to visualise the spatial distribution of focus across the image.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    rows, cols = grid
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        y0, y1 = r * h // rows, (r + 1) * h // rows
        for c in range(cols):
            x0, x1 = c * w // cols, (c + 1) * w // cols
            patch = gray[y0:y1, x0:x1]
            if patch.size:
                out[r, c] = cv2.Laplacian(patch, cv2.CV_64F).var()
    return out


def quality_warnings(q: dict[str, float]) -> list[str]:
    """Heuristic warnings to flag a low-quality input image."""
    warnings: list[str] = []
    if q["laplacian_variance"] < 100:
        warnings.append("Image may be out of focus (Laplacian variance < 100).")
    if q["dark_clip_pct"] > 1.0:
        warnings.append("Significant dark clipping (shadows lost).")
    if q["bright_clip_pct"] > 1.0:
        warnings.append("Significant bright clipping (highlights lost).")
    if q["std_gray"] < 25:
        warnings.append("Low contrast (std_gray < 25).")
    return warnings
