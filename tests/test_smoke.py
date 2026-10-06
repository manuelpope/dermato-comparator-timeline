"""Smoke tests for the simplified Athenas comparison pipeline.

Run with: ``uv run pytest``
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from athenas import background, cluster, comparison, enhance, io, report, segmentation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Pin a seed so random-image based assertions are reproducible across runs.
np.random.seed(0)


# ---------- io ------------------------------------------------------------

def test_read_rgb_raises_for_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Could not read image"):
        io.read_rgb(tmp_path / "nope.jpg")


def test_read_rgb_decodes_heic(tmp_path):
    """Write a small HEIC file via pillow-heif and confirm the pipeline
    can decode it as RGB uint8."""
    from PIL import Image
    from pillow_heif import register_heif_opener
    register_heif_opener()  # idempotent, but harmless to repeat

    rgb = (np.random.rand(64, 48, 3) * 255).astype(np.uint8)
    heic_path = tmp_path / "sample.heic"
    Image.fromarray(rgb).save(heic_path, format="HEIF")

    out = io.read_rgb(heic_path)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[-1] == 3
    assert out.shape[0] > 0 and out.shape[1] > 0


# ---------- background ----------------------------------------------------

def test_clean_binary_idempotent_on_full_mask():
    mask = np.ones((50, 50), dtype=np.uint8) * 255
    out = background.clean_binary(mask)
    assert (out == mask).all()


def test_apply_mask_zeros_background():
    rgb = np.full((20, 20, 3), 200, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    out = background.apply_mask(rgb, mask)
    assert (out[0:5, :] == 0).all()
    assert (out[15:, :] == 0).all()
    assert (out[5:15, 5:15] == 200).all()


def test_fill_background_white_replaces_only_background():
    rgb = np.full((20, 20, 3), 100, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    out = background.fill_background_white(rgb, mask)
    # Foreground kept, background replaced with 255.
    assert (out[5:15, 5:15] == 100).all()
    assert (out[0:5, :] == 255).all()
    assert (out[15:, 5:15] == 255).all()


def test_binarise_alpha_threshold():
    """Pixels with alpha < threshold → 0; ≥ threshold → 255."""
    alpha = np.array(
        [[0, 10, 31, 32, 33, 100, 200, 255]], dtype=np.uint8,
    )
    out = background._binarise_alpha(alpha, threshold=32)
    assert out.dtype == np.uint8
    # Pixels 0, 10, 31 are below 32 → 0
    assert (out[0, 0:3] == 0).all()
    # Pixels 32..255 are at or above 32 → 255
    assert (out[0, 3:] == 255).all()


def test_binarise_alpha_accepts_float():
    """A float alpha channel (e.g. 0..1 from a model) is clipped to 0..255.

    rembg can also emit a 0..1 float alpha; we just need to make sure the
    binarisation respects the threshold and produces a uint8 output.
    """
    alpha = np.array([[0.0, 127.0, 128.0, 255.0]], dtype=np.float32)
    out = background._binarise_alpha(alpha, threshold=128)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0    # below threshold
    assert out[0, 1] == 0    # below threshold
    assert out[0, 2] == 255  # at threshold
    assert out[0, 3] == 255  # above threshold


def test_remove_background_rembg_raises_when_missing():
    """If rembg isn't installed, the function raises a clear RuntimeError.

    We simulate the missing dependency by pointing the import machinery
    at a stub that raises ImportError.
    """
    import sys
    saved = sys.modules.pop("rembg", None)
    sys.modules["rembg"] = type(sys)("rembg")  # empty stub
    try:
        with pytest.raises(RuntimeError, match="rembg is not installed"):
            background.remove_background_rembg(np.zeros((10, 10, 3), dtype=np.uint8))
    finally:
        if saved is not None:
            sys.modules["rembg"] = saved
        else:
            sys.modules.pop("rembg", None)


# ---------- enhance -------------------------------------------------------

def test_to_grayscale_shape_and_dtype():
    rgb = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
    gray = enhance.to_grayscale(rgb)
    assert gray.shape == (40, 40)
    assert gray.dtype == np.uint8


def test_clahe_enhance_in_uint8_range():
    rgb = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
    gray = enhance.to_grayscale(rgb)
    out = enhance.clahe_enhance(gray)
    assert out.dtype == np.uint8
    assert out.shape == gray.shape


def test_unsharp_mask_in_uint8_range():
    gray = np.random.randint(0, 256, (40, 40), dtype=np.uint8)
    out = enhance.unsharp_mask(gray, amount=1.0, radius=3)
    assert out.dtype == np.uint8
    assert out.shape == gray.shape


def test_enhance_bw_returns_grayscale():
    rgb = np.random.randint(0, 256, (60, 60, 3), dtype=np.uint8)
    out = enhance.enhance_bw(rgb)
    assert out.ndim == 2
    assert out.dtype == np.uint8


# ---------- segmentation --------------------------------------------------

def test_hair_structure_mask_returns_uint8_mask_and_params():
    rgb = np.random.randint(0, 256, (60, 60, 3), dtype=np.uint8)
    head = np.zeros((60, 60), dtype=np.uint8)
    head[10:50, 10:50] = 255
    mask, params = segmentation.hair_structure_mask(rgb, head)
    assert mask.dtype == np.uint8
    assert mask.shape == rgb.shape[:2]
    assert set(params.keys()) == {"t_dark", "t_grad"}


# ---------- cluster -------------------------------------------------------

def test_numpy_kmeans_returns_k_centers():
    X = np.random.rand(500, 3).astype(np.float32)
    labels, centers = cluster.numpy_kmeans(X, k=4, iterations=20)
    assert centers.shape == (4, 3)
    assert labels.shape == (500,)
    assert set(labels.tolist()).issubset({0, 1, 2, 3})


def test_cluster_lab_shape_and_palette_size():
    rgb = np.random.randint(0, 256, (80, 60, 3), dtype=np.uint8)
    res = cluster.cluster_lab(rgb, k=3)
    assert res.rgb.shape == rgb.shape
    assert res.rgb.dtype == np.uint8
    # Only the k palette colours should appear.
    flat = res.rgb.reshape(-1, 3)
    unique = np.unique(flat, axis=0)
    assert len(unique) <= 3
    # Proportions must sum to 1.
    assert abs(sum(res.proportions) - 1.0) < 1e-6
    # Labels must have the right shape and dtype.
    assert res.labels.shape == rgb.shape[:2]


def test_density_heatmap_returns_uint8_rgb():
    rgb = np.random.randint(0, 256, (80, 60, 3), dtype=np.uint8)
    heat = cluster.density_heatmap(rgb, k=3, window=15)
    assert heat.shape == rgb.shape
    assert heat.dtype == np.uint8


# ---------- comparison ---------------------------------------------------

def test_fit_to_canvas_preserves_aspect_ratio():
    rgb = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)
    out = comparison.fit_to_canvas(rgb, target_h=100)
    assert out.shape[0] == 100
    assert out.shape[1] == 50  # aspect ratio preserved


def test_pad_to_canvas_centers_and_pads_with_white():
    rgb = np.full((10, 20, 3), 100, dtype=np.uint8)
    out = comparison.pad_to_canvas(rgb, (30, 40))
    assert out.shape == (30, 40, 3)
    # Original 10x20 block centered → at y=10..20, x=10..30.
    assert (out[10:20, 10:30] == 100).all()
    # The padding zones are white (255).
    assert (out[0:10, :] == 255).all()
    assert (out[20:, :] == 255).all()
    assert (out[:, 0:10] == 255).all()
    assert (out[:, 30:] == 255).all()


def test_pad_to_canvas_no_op_when_shape_matches():
    rgb = np.full((30, 40, 3), 100, dtype=np.uint8)
    out = comparison.pad_to_canvas(rgb, (30, 40))
    np.testing.assert_array_equal(out, rgb)


def test_head_bbox_returns_none_for_empty_mask():
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert comparison.head_bbox(mask) is None


def test_head_bbox_includes_padding():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # 20×20 square centred
    x, y, w, h = comparison.head_bbox(mask, padding_frac=0.10)
    assert w > 20 and h > 20
    # Bbox should still be centred on the mask
    assert (x + w / 2) == pytest.approx(50, abs=1)
    assert (y + h / 2) == pytest.approx(50, abs=1)


def test_head_bbox_stops_at_neck_not_shoulders():
    """Simulate a person bent forward: wide head at top, narrow neck, wide
    shoulders below. The bbox should stop at the neck, not extend into the
    shoulders.
    """
    mask = np.zeros((200, 100), dtype=np.uint8)
    # Head: rows 10–50, full width (90 px wide)
    mask[10:50, 5:95] = 255
    # Neck: rows 50–80, narrow (30 px wide centred)
    mask[50:80, 35:65] = 255
    # Shoulders/body: rows 80–180, full width again
    mask[80:180, 5:95] = 255

    x, y, w, h = comparison.head_bbox(mask, padding_frac=0.0)
    # The bbox should stop before the shoulders start.
    assert y + h <= 80  # not in the body region
    assert y + h >= 50  # but covers the head fully


def test_head_bbox_handles_wispy_top():
    """Top of head has sparse wispy hair strands. The detector must not
    collapse the bbox to a 1-pixel strip because of those sparse pixels.
    """
    mask = np.zeros((300, 200), dtype=np.uint8)
    # Wispy hair at the top — only a few pixels per row.
    for y in range(10, 30):
        xs = np.linspace(20, 180, num=3, dtype=int)
        mask[y, xs] = 255
    # Dense hair/head below — full width.
    mask[30:180, 20:180] = 255
    # Shoulders wide again below.
    mask[180:280, 20:180] = 255

    _, _, _, h = comparison.head_bbox(mask, padding_frac=0.0)
    # The bbox should NOT collapse to 1 px.
    assert h > 50


def test_head_crop_resize_targets_same_height():
    rgb = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
    mask = np.zeros((300, 200), dtype=np.uint8)
    mask[80:220, 60:140] = 255
    bbox = comparison.head_bbox(mask, padding_frac=0.0)
    out = comparison.head_crop_resize(rgb, bbox, target_h=200)
    assert out.shape[0] == 200
    # Width must respect the source bbox aspect ratio
    assert out.shape[1] == int(round(80 * (200 / 140)))


def test_make_overlay_zero_mask_equals_input():
    rgb = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    out = comparison.make_overlay(rgb, mask, alpha=0.5)
    np.testing.assert_array_equal(out, rgb)


def test_make_overlay_full_mask_overlays_green():
    rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
    mask = np.ones((20, 20), dtype=np.uint8) * 255
    out = comparison.make_overlay(rgb, mask, alpha=1.0)
    # With alpha=1.0 the overlay colour (green) wins completely.
    assert (out[..., 1] == 255).all()


# ---------- report -------------------------------------------------------

def _dummy_figure(label: str) -> object:
    """Tiny in-memory matplotlib Figure used as a stand-in for a stage."""
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=12)
    ax.axis("off")
    fig.canvas.draw()
    return fig


def test_make_report_writes_pdf(tmp_path):

    from athenas.pipeline import PipelineResult

    result = PipelineResult(
        original=_dummy_figure("stage1"),
        no_background=_dummy_figure("stage2"),
        bw_enhanced=_dummy_figure("stage3"),
        cluster=_dummy_figure("stage4"),
        scalp_percent_a=12.3,
        scalp_percent_b=45.6,
    )
    out_pdf = tmp_path / "report.pdf"
    returned = report.make_report(result, out_pdf)

    assert returned == out_pdf
    assert out_pdf.exists()
    # Sanity: a non-empty 2-page PDF is several kilobytes.
    assert out_pdf.stat().st_size > 1500


def test_fig_to_rgb_round_trip():
    """fig_to_rgb must produce an RGB uint8 array from an in-memory Figure."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "x", ha="center")
    ax.axis("off")
    fig.canvas.draw()
    arr = comparison.fig_to_rgb(fig)
    assert arr.dtype == np.uint8
    assert arr.ndim == 3 and arr.shape[-1] == 3
    assert arr.shape[0] > 0 and arr.shape[1] > 0
    plt.close(fig)
