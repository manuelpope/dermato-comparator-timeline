"""Side-by-side comparison figures (A vs B) — one per processing stage.

Both panels are forced to the same display size (default 800 px tall) so the
two photos are directly comparable, and they are placed with negligible
horizontal gap so the eye can scan across the boundary.

MVP2: figures are returned as in-memory ``matplotlib.figure.Figure`` objects —
no PNGs are ever written to disk. Callers (the report module) take care of
composing them into a final PDF.
"""

from __future__ import annotations

import io

import cv2
import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

# Display width of each panel in pixels; height is derived from aspect ratio.
DEFAULT_TARGET_HEIGHT: int = 800


def fit_to_canvas(rgb: np.ndarray, target_h: int = DEFAULT_TARGET_HEIGHT) -> np.ndarray:
    """Resize ``rgb`` so its height is exactly ``target_h`` px, preserving AR."""
    h, w = rgb.shape[:2]
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(rgb, (new_w, target_h), interpolation=cv2.INTER_AREA)


def pad_to_canvas(
    rgb: np.ndarray, target_shape: tuple[int, int],
) -> np.ndarray:
    """Pad ``rgb`` with white (255) so it matches ``target_shape`` (h, w).

    The input is **centered** in the new canvas. Returns ``rgb`` unchanged
    when the shape already matches. Used by the pipeline to force both A
    and B heads into the same canvas so the side-by-side comparison has
    consistent visual size regardless of capture distance.
    """
    h, w = rgb.shape[:2]
    th, tw = target_shape
    if h == th and w == tw:
        return rgb
    canvas = np.full((th, tw) + rgb.shape[2:], 255, dtype=rgb.dtype)
    y0 = (th - h) // 2
    x0 = (tw - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgb
    return canvas


def head_bbox(
    mask: np.ndarray,
    padding_frac: float = 0.45,
    cutoff_ratio: float = 0.65,
) -> tuple[int, int, int, int] | None:
    """Bounding box of the **head** (hair + face) — excludes shoulders/body.

    GrabCut labels anything non-background as foreground, which means the
    mask typically extends down to the torso / clothing. We don't want
    that — for trichology the head must be cropped tightly so the buzo /
    shirt doesn't pollute the comparison.

    The algorithm is **pose-agnostic** — it should work for upright,
    side-back, and bent-forward patients without any per-photo tuning.
    Strategy:

    1. Take the full foreground bbox from the mask.
    2. Compute mask-width per row inside that bbox.
    3. Skip the top 12 % of rows (the topmost wisps of hair are sparse —
       counting them as "the head" would underestimate the reference
       width and make the algorithm collapse the bbox to a 1-pixel
       strip). 12 % is generous so the forehead / face framing is kept.
    4. The reference head-width is the max width in the remaining top
       section.
    5. Find the first y-row where the silhouette narrows to less than
       45 % of the reference (looser than the original 60 % — a real
       neck pinch is closer to half-width, so 0.45 catches the neck
       without mistaking ear hair or a bent chin for it). If that
       narrowing is in the upper ``cutoff_ratio`` of the bbox, treat it
       as the neck transition and crop there. Otherwise (or when no
       narrowing is found) crop to ``cutoff_ratio`` of the bbox — the
       head always sits in the upper portion of the frame for any pose.
    6. Apply a small padding around the result.

    Returns ``(x, y, w, h)`` or ``None`` when the mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0_init, x1_init = int(xs.min()), int(xs.max())
    y0_init, y1_init = int(ys.min()), int(ys.max())
    bbox_h = y1_init - y0_init

    if bbox_h < 5:
        y1_new = y1_init
    else:
        widths = np.zeros(bbox_h + 1, dtype=np.int64)
        for i, y in enumerate(range(y0_init, y1_init + 1)):
            widths[i] = int(mask[y, x0_init:x1_init + 1].sum() // 255)

        # Skip only the top 10 % — generous so we don't crop the forehead
        # when the mask is already tight.
        search_start = max(1, int(len(widths) * 0.10))

        head_ref_w = int(widths[search_start:].max())
        # 0.40 catches a real neck (it pinches to ~50 %) while ignoring
        # side burns / ear hair (which stay close to the head width).
        threshold = max(1, int(round(head_ref_w * 0.40)))

        # Cap = "the head is always in the upper X % of the frame".
        head_cap = y0_init + int(round(bbox_h * cutoff_ratio))

        below = np.where(widths[search_start:] < threshold)[0]
        if len(below) > 0:
            narrow_idx = search_start + int(below[0])
            # Trust the narrowing only if it appears within the cap.
            # Beyond the cap, any narrowing is waist / leg taper, not neck.
            if narrow_idx < bbox_h * cutoff_ratio:
                y1_new = y0_init + narrow_idx
            else:
                y1_new = head_cap
        elif bbox_h > 300:
            y1_new = head_cap
        else:
            y1_new = y1_init

        # Hard clamp: head never extends past the cap when bbox is big.
        if bbox_h > 300 and y1_new > head_cap:
            y1_new = head_cap

    h, w = mask.shape[:2]
    pad_x = int(round((x1_init - x0_init) * padding_frac))
    pad_y = int(round((y1_new - y0_init) * padding_frac))
    x0 = max(0, x0_init - pad_x)
    x1 = min(w, x1_init + pad_x + 1)
    y0 = max(0, y0_init - pad_y)
    y1 = min(h, y1_new + pad_y + 1)
    return (x0, y0, x1 - x0, y1 - y0)


def head_crop_resize(
    rgb: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    target_h: int = DEFAULT_TARGET_HEIGHT,
) -> np.ndarray:
    """Crop ``rgb`` to ``bbox`` and resize so the result is ``target_h`` tall.

    This is the key to a fair head-to-head comparison: after GrabCut, both
    photos' heads are brought to the *same display height*, so any
    difference in zoom/distance is normalised away before the comparison
    is rendered.
    """
    if bbox is None:
        return fit_to_canvas(rgb, target_h=target_h)
    x, y, w, h = bbox
    cropped = rgb[y:y + h, x:x + w]
    return fit_to_canvas(cropped, target_h=target_h)


def make_comparison(
    img_a: np.ndarray,
    img_b: np.ndarray,
    title: str,
    label_a: str = "Baseline (A)",
    label_b: str = "Follow-up (B)",
    cmap: str | None = None,
    dpi: int = 150,
    target_h: int = DEFAULT_TARGET_HEIGHT,
    with_suptitles: bool = True,
) -> Figure:
    """Build a 1 × 2 side-by-side comparison figure of A vs B and return it.

    Both inputs are resized to the same display height with ``fit_to_canvas``
    before plotting, and the subplots are glued together with almost no gap
    (``wspace ≈ 0.01``) so the boundary is easy to scan across.

    When ``with_suptitles`` is False the figure / per-panel titles are
    suppressed — useful when the figure will be embedded into a parent
    Figure that already provides a header (the report module).

    Nothing is written to disk — the Figure lives until ``plt.close`` is
    called by the caller.
    """
    a = fit_to_canvas(img_a, target_h=target_h)
    b = fit_to_canvas(img_b, target_h=target_h)

    # Figure width derived from the *larger* of the two panels.
    panel_w = max(a.shape[1], b.shape[1]) / dpi
    fig_w = panel_w * 2 + 0.4
    fig_h = target_h / dpi + 0.6

    fig, ax = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.01, "hspace": 0.01, "width_ratios": [a.shape[1], b.shape[1]]},
    )
    ax[0].imshow(a, cmap=cmap)
    ax[1].imshow(b, cmap=cmap)
    if with_suptitles:
        ax[0].set_title(label_a, fontsize=12, weight="bold")
        ax[1].set_title(label_b, fontsize=12, weight="bold")
        fig.suptitle(title, fontsize=14, weight="bold", y=0.98)
    ax[0].axis("off")
    ax[1].axis("off")
    fig.canvas.draw()  # render so canvas-to-buffer has pixels
    return fig


def fig_to_rgb(fig: Figure, dpi: int = 150) -> np.ndarray:
    """Render a Figure to an RGB uint8 array, no disk round-trip."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    buf.close()
    return img


def make_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Return a green-tinted overlay of ``mask`` on top of ``rgb``."""
    out = rgb.copy().astype(np.float32)
    overlay = out.copy()
    overlay[mask > 0] = [0, 255, 0]
    return np.clip((1 - alpha) * out + alpha * overlay, 0, 255).astype(np.uint8)
