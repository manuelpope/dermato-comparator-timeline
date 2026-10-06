"""Unit tests for the dermatology filter recipes.

Each test feeds a small random RGB image through one filter and checks
shape, dtype, and value range. These tests LOCK the contracts the PDF
builder depends on — change a filter signature and the tests will catch it.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.dermatology.filters import (
    LesionFilters,
    filter_bw_contrast,
    filter_clahe,
    filter_color_contrast,
    filter_scale,
    parallel_apply_filters,
)


@pytest.fixture
def rgb_small() -> np.ndarray:
    """A tiny random uint8 RGB image."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, size=(80, 60, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Single-filter contract tests
# ---------------------------------------------------------------------------


def test_filter_clahe_returns_rgb_uint8(rgb_small):
    out = filter_clahe(rgb_small)
    assert out.dtype == np.uint8
    assert out.ndim == 3
    assert out.shape[2] == 3
    assert out.shape[:2] == rgb_small.shape[:2]
    assert out.min() >= 0 and out.max() <= 255


def test_filter_bw_contrast_returns_gray_uint8(rgb_small):
    out = filter_bw_contrast(rgb_small)
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.shape == rgb_small.shape[:2]
    assert out.min() >= 0 and out.max() <= 255


def test_filter_color_contrast_returns_rgb_uint8(rgb_small):
    out = filter_color_contrast(rgb_small)
    assert out.dtype == np.uint8
    assert out.ndim == 3
    assert out.shape[2] == 3
    assert out.shape[:2] == rgb_small.shape[:2]
    assert out.min() >= 0 and out.max() <= 255


def test_filter_scale_resizes_to_target(rgb_small):
    out = filter_scale(rgb_small, target_h=40)
    assert out.dtype == np.uint8
    assert out.ndim == 3
    assert out.shape[0] == 40
    # AR preserved (allow ±1 px rounding).
    assert abs(out.shape[1] - round(60 * 40 / 80)) <= 1


# ---------------------------------------------------------------------------
# Orchestrator test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_apply_filters_returns_all_four(rgb_small):
    result = await parallel_apply_filters(
        rgb_small,
        target_height=40,
        clahe_clip_limit=2.0,
        saturation_boost=1.2,
    )
    assert isinstance(result, LesionFilters)
    assert result.clahe.ndim == 3
    assert result.bw.ndim == 2
    assert result.color.ndim == 3
    assert result.normalised.shape[0] == 40