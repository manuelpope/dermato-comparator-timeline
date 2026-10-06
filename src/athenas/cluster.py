"""Pure-NumPy K-means clustering in Lab color space.

Used by stage 4 of the pipeline to produce a high-contrast structural map
(skin / hair / highlights) where the **proportion of "skin" pixels** is the
direct visual proxy for thinning.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Display palette chosen for max contrast against a trichology photo.
# The "skin / scalp" slot uses coral instead of natural skin tone so the
# exposed-scalp regions POP visually — that is the direct cue for thinning.
#   index 0 (darkest)    → hair shafts
#   index 1 (mid)        → scalp / skin (CATCHES THE EYE)
#   index 2 (lightest)   → highlights / bright reflections
_DISPLAY_PALETTE: np.ndarray = np.array(
    [
        (25, 15, 10),     # hair       — near-black
        (240, 150, 130),  # skin       — coral, signals exposed area
        (250, 240, 225),  # highlights — off-white cream
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class ClusterResult:
    """Outcome of :func:`cluster_lab`."""

    rgb: np.ndarray            # colourised RGB image (H×W×3, uint8)
    labels: np.ndarray         # 2-D label map (H×W, int64), sorted by ascending L
    proportions: list[float]   # proportion per cluster, sorted by ascending L


def numpy_kmeans(
    X: np.ndarray,
    k: int = 3,
    iterations: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm. Returns ``(labels, centers)``."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    centers = X[idx].copy().astype(np.float32)

    for _ in range(iterations):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            pts = X[labels == j]
            if len(pts):
                new_centers[j] = pts.mean(axis=0)
        if np.allclose(new_centers, centers, atol=1e-3):
            break
        centers = new_centers
    return labels, centers


def cluster_lab(
    rgb: np.ndarray,
    k: int = 3,
    seed: int = 42,
) -> ClusterResult:
    """Cluster pixels in Lab, sorted by ascending L (luminance).

    The darkest cluster is always painted as "hair", the mid cluster as
    "skin" (coral), the lightest as "highlights". This means the *area of
    coral* on the result IS the visible-scalp area, and its proportion
    gives a direct thinning score.
    """
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)

    labels, centers = numpy_kmeans(flat, k=k, seed=seed)

    # Re-order clusters by ascending L so palette index is consistent.
    order = np.argsort(centers[:, 0])
    remap = np.empty(k, dtype=np.int64)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id
    sorted_flat = remap[labels]
    sorted_2d = sorted_flat.reshape(h, w)

    palette = _DISPLAY_PALETTE[:k]
    out_rgb = palette[sorted_flat].reshape(h, w, 3)

    proportions = [(sorted_2d == i).mean() for i in range(k)]

    return ClusterResult(rgb=out_rgb, labels=sorted_2d, proportions=proportions)


def density_heatmap(
    rgb: np.ndarray,
    k: int = 3,
    seed: int = 42,
    window: int = 31,
) -> np.ndarray:
    """Return a turbo-colormap heatmap of local "skin/scalp" density.

    Every pixel gets a value in [0, 1] equal to the proportion of *skin*
    pixels (the mid-L cluster) inside a ``window × window`` Gaussian-weighted
    neighbourhood. Hot colours (orange/yellow) = the calibration curve: **the
    redder, the more "balder"**. This makes bald patches and diffuse
    low-density areas immediately visible, even before reading any numbers.
    """
    res = cluster_lab(rgb, k=k, seed=seed)
    # Build a binary "is skin" image (cluster index 1 for k ≥ 3).
    skin = (res.labels == 1).astype(np.float32)

    sigma = max(1.0, window / 6.0)
    blurred = cv2.GaussianBlur(skin, (window, window), sigma)

    # Normalise per image so the most-skin window in THIS photo maps to 1.
    hi = float(blurred.max())
    if hi > 1e-6:
        normalised = np.clip(blurred / hi, 0.0, 1.0)
    else:
        normalised = blurred

    heat_u8 = (normalised * 255).astype(np.uint8)
    return cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
