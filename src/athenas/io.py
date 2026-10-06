"""Image I/O helpers — minimal, used by the comparison pipeline.

Supports JPG / PNG via OpenCV (fast path) and HEIC / HEIF via
``pillow-heif`` (registered as a Pillow plugin at import time).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# HEIC/HEIF need an explicit plugin registration with Pillow. Imported
# for side-effect: ``pillow_heif.register_heif_opener()`` patches Pillow's
# plugin loader so ``Image.open(...)`` can decode .heic / .heif files.
try:
    from pillow_heif import register_heif_opener  # type: ignore[import-not-found]

    register_heif_opener()
except ImportError:  # pragma: no cover — pillow-heif is a hard dep, this is a safety net
    _HEIF_OK = False
else:
    _HEIF_OK = True

_HEIF_EXTS: frozenset[str] = frozenset({".heic", ".heif"})


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image as an ``uint8`` RGB ``ndarray``.

    Accepts JPG / PNG / WEBP / BMP / TIFF (via OpenCV) and HEIC / HEIF
    (via Pillow + pillow-heif).

    Raises
    ------
    ValueError
        If neither decoder can open the file.
    """
    p = Path(path)
    if p.suffix.lower() in _HEIF_EXTS:
        return _read_via_pillow(p)
    # Fast path for the common formats.
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # cv2 couldn't decode it — try Pillow as a last resort (handles HEIC
    # if pillow-heif is installed, plus a few other exotic formats).
    try:
        return _read_via_pillow(p)
    except Exception as exc:
        raise ValueError(f"Could not read image at: {path} ({exc})") from exc


def _read_via_pillow(p: Path) -> np.ndarray:
    """Decode ``p`` with Pillow and return an ``uint8`` RGB array."""
    with Image.open(p) as im:
        im.load()
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    """Save an RGB ``ndarray`` as a PNG/JPG image."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def save_gray(path: str | Path, arr: np.ndarray) -> None:
    """Save a 2-D grayscale ``ndarray`` as a PNG image."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)
