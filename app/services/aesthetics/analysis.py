"""Geometry + metrics of the aesthetics comparison.

This module holds the *normalisation* logic — the part that makes the
before/after pair comparable — plus the per-region metrics the PDF reports.

Normalisation, step by step
----------------------------
1. **Similarity alignment** — the "after" photo is warped onto the "before"
   frame (rotation + uniform scale + translation), so both faces have the
   same head size and the eyes sit on the same line.
2. **Shared crop** — the union of both face ovals (after alignment) is
   cropped with *one* rectangle, so nothing important is cut off.
3. **Common resize** — both crops are resized to the same height, which
   makes the head size identical in the report.

Nothing here touches the disk or the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from athenas.face_expression.alignment import (
    AlignmentTransform,
    ProtocolCheck,
)
from athenas.face_expression.face_detection import (
    FACE_OVAL_INDICES,
    FaceLandmarks,
    FacePose,
)

#: Human-readable (Spanish) labels for the wrinkle regions.
REGION_LABELS: dict[str, str] = {
    "frontal": "Frontal",
    "glabellar": "Glabelar",
    "periorbital_L": "Periorbital izq.",
    "periorbital_R": "Periorbital der.",
    "nasolabial_L": "Nasogeniano izq.",
    "nasolabial_R": "Nasogeniano der.",
    "perioral": "Perioral",
}


@dataclass(frozen=True)
class RegionDelta:
    """Change of the wrinkle density of one facial region.

    ``delta`` is ``after − before`` (negative = fewer lines after the
    botox session). ``delta_pct`` is the relative change; it is ``None``
    when the "before" density is zero (percentage is meaningless there).
    """

    region: str
    label: str
    density_before: float
    density_after: float
    delta: float
    delta_pct: float | None


@dataclass(frozen=True)
class AestheticsAnalysis:
    """Everything the PDF report needs about one before/after pair."""

    views: object  # app.services.aesthetics.filters.AestheticsViews
    region_deltas: list[RegionDelta]
    protocol_checks: list[ProtocolCheck]
    pose_before: FacePose
    pose_after: FacePose
    alignment: AlignmentTransform
    target_height: int


def face_bbox(
    landmarks_xy: np.ndarray,
    *,
    padding_frac: float = 0.25,
) -> tuple[float, float, float, float]:
    """Bounding box ``(x, y, w, h)`` of the face oval with padding."""
    pts = landmarks_xy[list(FACE_OVAL_INDICES)]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w = float(x1 - x0)
    h = float(y1 - y0)
    pad = padding_frac * max(w, h)
    return float(x0 - pad), float(y0 - pad), w + 2 * pad, h + 2 * pad


def shared_face_crop(
    landmarks_before_xy: np.ndarray,
    landmarks_after_xy: np.ndarray,
    image_shape_hw: tuple[int, int],
    *,
    padding_frac: float = 0.25,
) -> tuple[int, int, int, int]:
    """One crop rectangle covering both (already aligned) faces.

    The rectangle is the union of the two face-oval boxes, clamped to the
    image, and is later used for *both* photos — that is what keeps the
    two heads exactly the same size in the report.
    """
    h, w = image_shape_hw
    boxes = [
        face_bbox(landmarks_before_xy, padding_frac=padding_frac),
        face_bbox(landmarks_after_xy, padding_frac=padding_frac),
    ]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)

    x0 = int(max(0, round(x0)))
    y0 = int(max(0, round(y0)))
    x1 = int(min(w, round(x1)))
    y1 = int(min(h, round(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "El rostro quedó fuera de la imagen al normalizar — "
            "no se puede comparar el par de fotos."
        )
    return x0, y0, x1 - x0, y1 - y0


def crop_and_resize(
    rgb: np.ndarray,
    rect: tuple[int, int, int, int],
    target_height: int,
) -> np.ndarray:
    """Crop ``rgb`` with ``rect`` and resize it to ``target_height``."""
    x, y, w, h = rect
    crop = rgb[y : y + h, x : x + w]
    if crop.size == 0:
        raise ValueError("El recorte normalizado quedó vacío.")
    scale = target_height / crop.shape[0]
    new_w = max(1, int(round(crop.shape[1] * scale)))
    return cv2.resize(
        crop, (new_w, target_height), interpolation=cv2.INTER_AREA,
    )


def landmarks_to_crop(
    landmarks_xy: np.ndarray,
    rect: tuple[int, int, int, int],
    target_height: int,
) -> np.ndarray:
    """Move landmarks into the cropped+resized frame.

    Same transform the image went through, so the region masks line up
    with the pixels of the normalised photo.
    """
    x, y, w, h = rect
    scale = target_height / h
    pts = np.asarray(landmarks_xy, dtype=np.float32).copy()
    pts[:, 0] = (pts[:, 0] - x) * scale
    pts[:, 1] = (pts[:, 1] - y) * scale
    return pts


def cropped_landmarks(
    landmarks: FaceLandmarks,
    rect: tuple[int, int, int, int],
    target_height: int,
    target_width: int,
) -> FaceLandmarks:
    """Rebuild a :class:`FaceLandmarks` in the cropped/resized frame."""
    pixels = landmarks_to_crop(landmarks.pixels_xy, rect, target_height)
    return FaceLandmarks(
        pixels_xy=pixels,
        normalised_xyz=landmarks.normalised_xyz,
        image_shape_hw=(target_height, target_width),
    )


def compute_region_deltas(
    before: dict[str, float],
    after: dict[str, float],
) -> list[RegionDelta]:
    """Compare the per-region wrinkle densities of both photos."""
    deltas: list[RegionDelta] = []
    for region, density_before in before.items():
        density_after = after.get(region, 0.0)
        delta = density_after - density_before
        delta_pct = (delta / density_before * 100.0) if density_before > 0 else None
        deltas.append(
            RegionDelta(
                region=region,
                label=REGION_LABELS.get(region, region),
                density_before=density_before,
                density_after=density_after,
                delta=delta,
                delta_pct=delta_pct,
            )
        )
    return deltas
