"""Background removal — supports two backends: rembg (default) and GrabCut.

Both backends share the same public contract: ``(mask, rgba)`` where
``mask`` is a ``(H, W) uint8`` 0/255 binary mask and ``rgba`` is a
``(H, W, 4) uint8`` image. Downstream code (``apply_mask``,
``fill_background_white``, ``head_bbox``) only depends on this contract
so the two backends are interchangeable via ``PipelineConfig.background_method``.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---- shared helpers ----------------------------------------------------

def clean_binary(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Morphological close+open to clean a binary mask."""
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _binarise_alpha(alpha: np.ndarray, threshold: int = 32) -> np.ndarray:
    """Convert a soft alpha channel (any dtype, 0..max) to a 0/255 mask.

    Pixels with alpha < threshold become background (0). All others are
    foreground (255). The lower the threshold, the more halo is kept.
    Extracted from :func:`remove_background_rembg` so it can be unit-tested
    without downloading a model.
    """
    if alpha.dtype != np.uint8:
        alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    return np.where(alpha >= threshold, 255, 0).astype(np.uint8)


def apply_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero-out every pixel where ``mask == 0``."""
    out = rgb.copy()
    out[mask == 0] = 0
    return out


def fill_background_white(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace background pixels with white instead of zeroing them.

    Useful before running k-means because a black background would
    otherwise be classified as the *darkest* cluster and steal the
    'hair' slot.
    """
    out = rgb.copy()
    out[mask == 0] = 255
    return out


# ---- backend 1: OpenCV GrabCut -----------------------------------------

def remove_background_grabcut(
    rgb: np.ndarray,
    rect_frac: tuple[float, float, float, float] = (0.08, 0.03, 0.84, 0.90),
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run GrabCut and return ``(mask, rgba)``.

    The rectangle is specified as fractions of the image size, so the
    same call works at any resolution. The largest connected component
    is kept, which protects against GrabCut leaking into a background
    corner. Offline — no model download.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    x = int(w * rect_frac[0])
    y = int(h * rect_frac[1])
    rw = int(w * rect_frac[2])
    rh = int(h * rect_frac[3])

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        bgr,
        mask,
        (x, y, rw, rh),
        bgd,
        fgd,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = clean_binary(fg, 5)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        fg = np.where(labels == best, 255, 0).astype(np.uint8)

    rgba = np.dstack([rgb, fg])
    return fg, rgba


# ---- backend 2: rembg (deep-learning portrait matting) ----------------

@lru_cache(maxsize=4)
def _rembg_session(model_name: str):
    """Return a cached rembg session for ``model_name`` (loads the model on
    first call, ~200 MB download). The cache is per-process."""
    try:
        from rembg import new_session
    except ImportError as exc:
        raise RuntimeError(
            "rembg is not installed. Run `uv add 'rembg[cpu]>=2.0'` or pass "
            "--bg-method grabcut to use the offline fallback."
        ) from exc
    log.info("Loading rembg model %r (first call may download it)", model_name)
    return new_session(model_name)


def remove_background_rembg(
    rgb: np.ndarray,
    model: str = "birefnet-portrait",
    alpha_threshold: int = 32,
    decontaminate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run rembg on ``rgb`` and return ``(mask, rgba)``.

    The default model ``birefnet-portrait`` is fine-tuned for human
    portraits and preserves wispy hair edges that GrabCut routinely
    drops. ``decontaminate`` removes the coloured halo that the model
    leaves on white-ish backgrounds.

    ``alpha_threshold`` is applied to the soft alpha channel: any pixel
    with ``alpha < threshold`` becomes background. Lower → more halo.
    """
    try:
        from rembg import remove
    except ImportError as exc:
        raise RuntimeError(
            "rembg is not installed. Run `uv add 'rembg[cpu]>=2.0'` or pass "
            "--bg-method grabcut to use the offline fallback."
        ) from exc

    session = _rembg_session(model)
    rgba_pil = remove(rgb, session=session, decontaminate=decontaminate)

    # ``remove`` returns an RGBA PIL image. Pull the alpha channel.
    rgba_arr = np.asarray(rgba_pil, dtype=np.uint8)
    if rgba_arr.shape[-1] == 4:
        alpha = rgba_arr[..., 3]
    else:
        # Defensive: if rembg ever returns RGB-only, treat non-zero as fg.
        alpha = np.where(rgba_arr.sum(axis=-1) > 0, 255, 0).astype(np.uint8)

    mask = _binarise_alpha(alpha, threshold=alpha_threshold)
    # Re-attach the original RGB so the rgba return matches the
    # GrabCut backend's contract (original colour, not the matted one).
    rgba = np.dstack([rgb, mask])
    return mask, rgba
