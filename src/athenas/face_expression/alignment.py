"""Similarity-transform alignment + protocol validation.

Given two sets of face landmarks (A = baseline, B = follow-up), we:

1. Estimate the 2×3 similarity transform mapping B's landmarks into A's
   coordinate frame. Similarity = uniform scale + rotation + translation
   (no shear, no perspective). We use ``cv2.estimateAffinePartial2D`` with
   five reliable points (iris centres + outer eye corners + nose tip) and
   ``cv2.LMEDS`` for robustness to one outlier landmark.
2. Apply the transform to image B with ``cv2.warpAffine``.
3. Validate the photo-capture protocol — if pose diff is too big, scale
   ratio too far from 1.0, or lighting differs a lot, we surface a
   ``ProtocolCheck`` with severity ``"info" | "warn" | "fail"``.

Public dataclasses
------------------
- :class:`AlignmentTransform`
- :class:`ProtocolCheck`
- :class:`ProtocolThresholds`

Public functions
----------------
- :func:`compute_alignment_transform`
- :func:`apply_alignment`
- :func:`transform_landmarks`
- :func:`validate_protocol`
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from athenas.face_expression.face_detection import (
    LEFT_IRIS_CENTER_INDEX,
    RIGHT_IRIS_CENTER_INDEX,
    FaceLandmarks,
    FacePose,
    compute_inter_pupillary_distance,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentTransform:
    """2×3 similarity transform mapping B's landmarks into A's frame.

    Attributes
    ----------
    matrix_2x3
        ``(2, 3)`` float64 array directly usable by ``cv2.warpAffine``.
    scale
        Uniform scale factor applied by the transform. ``1.0`` = no zoom
        change; ``1.05`` means B was 5 % smaller than A.
    rotation_deg
        In-plane rotation in degrees. Positive = CCW.
    translation_xy
        ``(tx, ty)`` translation in pixels of A's frame.
    inter_pupillary_distance_a_px
        IPD of A in pixels (the reference distance).
    inter_pupillary_distance_b_px
        IPD of B in pixels before alignment.
    ipd_ratio_b_over_a
        ``IPD_B / IPD_A`` — the "true" scale ratio (the matrix scale
        is also a scale, but IPD is more trustworthy because it is the
        most reliably localised feature in MediaPipe).
    pose_diff
        ``B - A`` in degrees.
    """

    matrix_2x3: np.ndarray
    scale: float
    rotation_deg: float
    translation_xy: tuple[float, float]
    inter_pupillary_distance_a_px: float
    inter_pupillary_distance_b_px: float
    ipd_ratio_b_over_a: float
    pose_diff: FacePose


@dataclass(frozen=True)
class ProtocolCheck:
    """One photo-capture protocol rule.

    The visual UI renders these with emojis: 🟢 ``info`` & passed, 🟡
    ``warn`` or ``fail``, 🔴 ``fail``.
    """

    name: str
    observed: float
    expected: float
    passed: bool
    severity: str     # "info" | "warn" | "fail"
    message: str


@dataclass(frozen=True)
class ProtocolThresholds:
    """Tunable thresholds — override any field to make the protocol stricter."""

    pose_diff_max_deg: float = 8.0
    scale_ratio_max_delta: float = 0.15        # |scale_ratio - 1.0|
    ipd_ratio_max_delta: float = 0.10          # |ipd_ratio - 1.0|
    lighting_mean_gray_delta: float = 30.0
    lighting_std_gray_delta: float = 25.0


DEFAULT_THRESHOLDS = ProtocolThresholds()


# ---------------------------------------------------------------------------
# 2D set of 5 points used to estimate the similarity transform.
# ---------------------------------------------------------------------------
# Iris centres (468 / 473) are the most accurately localised points in the
# MediaPipe mesh — they only exist with refine_landmarks=True. Outer eye
# corners (33 / 263) and nose tip (1) round out a robust 5-point set that
# stays on the central face even at moderate angles.
_ALIGNMENT_2D_INDICES: tuple[int, ...] = (
    LEFT_IRIS_CENTER_INDEX,
    RIGHT_IRIS_CENTER_INDEX,
    33,    # left eye outer corner
    263,   # right eye outer corner
    1,     # nose tip
)

# Fallback used when iris landmarks are absent (refine_landmarks=False).
_ALIGNMENT_FALLBACK_INDICES: tuple[int, ...] = (
    33, 263, 1,  # outer eye corners + nose tip
    61, 291,     # mouth corners — give us a 5th pair of points
)


# ---------------------------------------------------------------------------
# Transform estimation
# ---------------------------------------------------------------------------


def _pick_alignment_indices(landmarks: FaceLandmarks) -> tuple[int, ...]:
    """Use iris-based 5-point set if available, else fallback."""
    if landmarks.normalised_xyz.shape[0] > RIGHT_IRIS_CENTER_INDEX:
        return _ALIGNMENT_2D_INDICES
    return _ALIGNMENT_FALLBACK_INDICES


def compute_alignment_transform(
    landmarks_a: FaceLandmarks,
    landmarks_b: FaceLandmarks,
    *,
    pose_a: FacePose | None = None,
    pose_b: FacePose | None = None,
) -> AlignmentTransform:
    """Estimate the similarity transform mapping B → A.

    Parameters
    ----------
    landmarks_a, landmarks_b
        Detected landmarks from :func:`face_detection.detect_landmarks`.
    pose_a, pose_b
        Optional pre-computed poses. If not provided, the IPD is used as the
        scale signal and the pose-diff is left as zero.

    Returns
    -------
    AlignmentTransform
        Contains the 2×3 matrix, scale, rotation, translation, IPDs, and the
        pose diff (B - A) when ``pose_a`` and ``pose_b`` were provided.

    Raises
    ------
    ValueError
        If ``cv2.estimateAffinePartial2D`` returns ``None`` — typically
        because the face in one of the photos is too much in profile.
    """
    indices = _pick_alignment_indices(landmarks_a)
    if indices != _pick_alignment_indices(landmarks_b):
        # Different refinement settings → fall back to the smaller set.
        indices = _ALIGNMENT_FALLBACK_INDICES

    pts_a = landmarks_a.pixels_xy[list(indices)]
    pts_b = landmarks_b.pixels_xy[list(indices)]

    matrix, inliers = cv2.estimateAffinePartial2D(
        pts_b, pts_a,
        method=cv2.LMEDS,
    )
    if matrix is None:
        raise ValueError(
            "Could not estimate alignment transform — face in one of the "
            "photos is too off-plane (profile / occluded). Re-capture the "
            "photo with a more frontal pose."
        )

    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    rotation_rad = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    rotation_deg = float(np.degrees(rotation_rad))
    translation_xy = (float(matrix[0, 2]), float(matrix[1, 2]))

    ipd_a = compute_inter_pupillary_distance(landmarks_a)
    ipd_b = compute_inter_pupillary_distance(landmarks_b)
    ipd_ratio = (ipd_b / ipd_a) if ipd_a > 0 else 1.0

    if pose_a is not None and pose_b is not None:
        pose_diff = FacePose(
            yaw_deg=pose_b.yaw_deg - pose_a.yaw_deg,
            pitch_deg=pose_b.pitch_deg - pose_a.pitch_deg,
            roll_deg=pose_b.roll_deg - pose_a.roll_deg,
        )
    else:
        pose_diff = FacePose(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0)

    return AlignmentTransform(
        matrix_2x3=matrix.astype(np.float32),
        scale=scale,
        rotation_deg=rotation_deg,
        translation_xy=translation_xy,
        inter_pupillary_distance_a_px=ipd_a,
        inter_pupillary_distance_b_px=ipd_b,
        ipd_ratio_b_over_a=ipd_ratio,
        pose_diff=pose_diff,
    )


# ---------------------------------------------------------------------------
# Apply transform
# ---------------------------------------------------------------------------


def apply_alignment(
    rgb_b: np.ndarray,
    transform: AlignmentTransform,
    *,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Warp image B into A's coordinate frame.

    The output canvas is the same size as ``rgb_b`` (we keep B's shape as
    the target so the caller can decide whether to crop or pad). Borders
    introduced by rotation/scale are filled with white.
    """
    h, w = rgb_b.shape[:2]
    return cv2.warpAffine(
        rgb_b,
        transform.matrix_2x3,
        (w, h),
        borderMode=border_mode,
        borderValue=border_value,
    )


def transform_landmarks(
    landmarks_xy: np.ndarray,
    transform: AlignmentTransform,
) -> np.ndarray:
    """Apply the similarity transform to a set of (x, y) points.

    Pure numpy — no MediaPipe or OpenCV calls.
    """
    pts = np.asarray(landmarks_xy, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.column_stack([pts, ones])
    return (transform.matrix_2x3 @ homo.T).T.astype(np.float32)


# ---------------------------------------------------------------------------
# Protocol validation
# ---------------------------------------------------------------------------


def _abs_deg(pose: FacePose) -> float:
    """Sum of absolute pose-axis diffs — coarse 'how different was this head?'."""
    return abs(pose.yaw_deg) + abs(pose.pitch_deg) + abs(pose.roll_deg)


def validate_protocol(
    pose_diff: FacePose,
    transform: AlignmentTransform,
    quality_a: dict[str, float],
    quality_b: dict[str, float],
    *,
    thresholds: ProtocolThresholds = DEFAULT_THRESHOLDS,
) -> list[ProtocolCheck]:
    """Run every photo-capture protocol rule and return the checks.

    Each rule is independent and the caller decides whether to surface a
    warning, fail the request, or both. The function **never** raises.
    """
    checks: list[ProtocolCheck] = []

    # 1. Pose diff — single combined metric + per-axis breakdown.
    abs_diff = _abs_deg(pose_diff)
    pose_passed = abs_diff <= thresholds.pose_diff_max_deg
    checks.append(ProtocolCheck(
        name="pose_diff",
        observed=round(abs_diff, 2),
        expected=0.0,
        passed=pose_passed,
        severity="info" if pose_passed else "warn",
        message=(
            f"Excelente ({abs_diff:.1f}° diferencia de pose)"
            if pose_passed
            else f"Diferencia de pose alta ({abs_diff:.1f}° > "
                 f"{thresholds.pose_diff_max_deg}°). Revisar antes de comparar."
        ),
    ))

    # 2. Scale ratio (IPD-based, the trustworthy signal).
    scale_delta = abs(transform.ipd_ratio_b_over_a - 1.0)
    scale_passed = scale_delta <= thresholds.scale_ratio_max_delta
    checks.append(ProtocolCheck(
        name="scale_ratio",
        observed=round(transform.ipd_ratio_b_over_a, 4),
        expected=1.0,
        passed=scale_passed,
        severity="info" if scale_passed else "warn",
        message=(
            f"Escala corregida ({scale_delta * 100:.1f}% diferencia)"
            if scale_passed
            else f"Escala muy diferente ({scale_delta * 100:.1f}% > "
                 f"{thresholds.scale_ratio_max_delta * 100:.1f}%). "
                 "Verificar distancia de captura."
        ),
    ))

    # 3. Lighting delta (mean gray).
    if "mean_gray" in quality_a and "mean_gray" in quality_b:
        mean_delta = abs(quality_a["mean_gray"] - quality_b["mean_gray"])
        light_passed = mean_delta <= thresholds.lighting_mean_gray_delta
        checks.append(ProtocolCheck(
            name="lighting_mean",
            observed=round(mean_delta, 2),
            expected=0.0,
            passed=light_passed,
            severity="info" if light_passed else "warn",
            message=(
                "Iluminación consistente"
                if light_passed
                else f"Iluminación diferente — mean_gray diff {mean_delta:.1f} "
                     f"(>{thresholds.lighting_mean_gray_delta:.0f})"
            ),
        ))

    # 4. Lighting delta (std gray / contrast).
    if "std_gray" in quality_a and "std_gray" in quality_b:
        std_delta = abs(quality_a["std_gray"] - quality_b["std_gray"])
        std_passed = std_delta <= thresholds.lighting_std_gray_delta
        checks.append(ProtocolCheck(
            name="lighting_std",
            observed=round(std_delta, 2),
            expected=0.0,
            passed=std_passed,
            severity="info" if std_passed else "warn",
            message=(
                "Contraste consistente"
                if std_passed
                else f"Contraste diferente — std_gray diff {std_delta:.1f} "
                     f"(>{thresholds.lighting_std_gray_delta:.0f})"
            ),
        ))

    return checks
