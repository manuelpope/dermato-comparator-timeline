"""Aesthetics orchestrator — normalise, filter and report a Botox pair.

Pipeline (synchronous, CPU-bound — runs in a worker thread):

1. Read both uploads into RGB arrays.
2. Detect the face + landmarks + head pose with MediaPipe.
3. Warp the "after" photo onto the "before" frame (similarity transform:
   same head size, eyes on the same line).
4. Crop **both** faces with one shared rectangle and resize them to the
   same height — this is the normalisation that makes the comparison
   trustworthy.
5. Detect the expression lines per region on both normalised faces and
   compare the densities (the actual botox effect).
6. Validate the capture protocol (pose / scale / lighting).
7. Render the PDF.

Steps 2–7 are dispatched to the stdlib thread pool via
``asyncio.to_thread`` so the FastAPI event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

from app.core.config import Settings
from app.services.aesthetics.analysis import (
    AestheticsAnalysis,
    compute_region_deltas,
    crop_and_resize,
    cropped_landmarks,
    shared_face_crop,
)
from app.services.aesthetics.filters import (
    AestheticsViews,
    build_change_map,
    build_face_view,
)
from app.services.pdf import build_aesthetics_pdf
from athenas.face_expression.alignment import (
    apply_alignment,
    compute_alignment_transform,
    transform_landmarks,
    validate_protocol,
)
from athenas.face_expression.face_detection import (
    FACE_OVAL_INDICES,
    FaceLandmarks,
    create_face_landmarker,
    detect_landmarks,
    estimate_pose,
)
from athenas.face_expression.wrinkle_detection import (
    build_region_mask,
    run_wrinkle_detection,
)
from athenas.io import read_rgb
from athenas.quality import quality_metrics

#: The temp file's extension is part of a path that ends up in the API
#: error message, so only a short alphanumeric extension is kept — markup
#: must never reach the temp file name.
_EXTENSION_RE = re.compile(r"^\.([A-Za-z0-9]{1,10})")


def _safe_suffix(filename: str | None) -> str:
    """Derive a harmless extension from the upload name.

    ``.jpg`` / ``.png`` / ``.heic`` survive untouched (the HEIC decoder is
    picked by extension); anything else is dropped, so the temp file name
    can never carry markup.
    """
    match = _EXTENSION_RE.match(Path(filename or "").suffix)
    return f".{match.group(1).lower()}" if match else ".bin"


async def _spool_to_tempfile(upload: UploadFile, tmp_dir: Path) -> Path:
    """Persist an UploadFile's bytes into a real on-disk file.

    :func:`athenas.io.read_rgb` expects a path (and registers the HEIC
    decoder on import), so writing to a request-scoped tempdir is the
    simplest, safest path.
    """
    fd, path = tempfile.mkstemp(dir=tmp_dir, suffix=_safe_suffix(upload.filename))
    try:
        while True:
            chunk = await upload.read(8192)
            if not chunk:
                break
            os.write(fd, chunk)
    finally:
        os.close(fd)
    return Path(path)


async def _read_one(upload: UploadFile, tmp_dir: Path) -> np.ndarray:
    """Read one upload into an RGB ndarray on the stdlib thread pool."""
    path = await _spool_to_tempfile(upload, tmp_dir)
    return await asyncio.to_thread(_read_rgb_flat, str(path))


def _read_rgb_flat(path: str) -> np.ndarray:
    """Read an image as RGB, compositing any alpha channel over white.

    :func:`athenas.io.read_rgb` drops the alpha channel, which turns a
    transparent background into black — ugly in the report and harmful
    for the change map. PNG screenshots and exported cut-outs are common
    in this domain, so we flatten them here (the shared helper stays
    untouched for the other endpoints).
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is not None and raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        bgr = raw[:, :, :3].astype(np.float32)
        flat = bgr * alpha + 255.0 * (1.0 - alpha)
        return cv2.cvtColor(flat.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return read_rgb(path)


def _require_face(landmarks: FaceLandmarks | None, which: str) -> FaceLandmarks:
    if landmarks is None:
        raise ValueError(
            f"No se detectó un rostro en la foto {which}. "
            "Repite la toma con el rostro de frente, bien iluminado y sin obstrucciones."
        )
    return landmarks


def run_analysis(
    rgb_before: np.ndarray,
    rgb_after: np.ndarray,
    *,
    target_height: int,
    padding_frac: float,
    wrinkle_threshold: float,
    wrinkle_kernel_length: int,
    wrinkle_min_line_length: int,
) -> AestheticsAnalysis:
    """Run the whole CPU-bound analysis and return the report payload."""
    landmarker = create_face_landmarker()

    lm_before = _require_face(
        detect_landmarks(landmarker, rgb_before), "anterior",
    )
    lm_after = _require_face(
        detect_landmarks(landmarker, rgb_after), "posterior",
    )
    pose_before = estimate_pose(landmarker, rgb_before, lm_before)
    pose_after = estimate_pose(landmarker, rgb_after, lm_after)

    # 1 — Align the "after" photo onto the "before" frame.
    transform = compute_alignment_transform(
        lm_before, lm_after, pose_a=pose_before, pose_b=pose_after,
    )
    aligned_after = apply_alignment(rgb_after, transform)
    aligned_after_pts = transform_landmarks(lm_after.pixels_xy, transform)
    lm_after_aligned = FaceLandmarks(
        pixels_xy=aligned_after_pts,
        normalised_xyz=lm_after.normalised_xyz,
        image_shape_hw=lm_after.image_shape_hw,
    )

    # 2 — One shared crop + one common height for both faces.
    rect = shared_face_crop(
        lm_before.pixels_xy,
        lm_after_aligned.pixels_xy,
        rgb_before.shape[:2],
        padding_frac=padding_frac,
    )
    before_norm = crop_and_resize(rgb_before, rect, target_height)
    after_norm = crop_and_resize(aligned_after, rect, target_height)

    lm_before_crop = cropped_landmarks(
        lm_before, rect, target_height, before_norm.shape[1],
    )
    lm_after_crop = cropped_landmarks(
        lm_after_aligned, rect, target_height, after_norm.shape[1],
    )

    # 3 — Expression lines on both normalised faces.
    wrinkles_before = run_wrinkle_detection(
        before_norm,
        lm_before_crop,
        threshold_percentile=wrinkle_threshold,
        kernel_length=wrinkle_kernel_length,
        min_line_length_px=wrinkle_min_line_length,
    )
    wrinkles_after = run_wrinkle_detection(
        after_norm,
        lm_after_crop,
        threshold_percentile=wrinkle_threshold,
        kernel_length=wrinkle_kernel_length,
        min_line_length_px=wrinkle_min_line_length,
    )

    # 4 — Metrics + protocol checks.
    deltas = compute_region_deltas(
        {name: r.density for name, r in wrinkles_before.per_region.items()},
        {name: r.density for name, r in wrinkles_after.per_region.items()},
    )
    checks = validate_protocol(
        transform.pose_diff,
        transform,
        quality_metrics(before_norm),
        quality_metrics(after_norm),
    )

    # The change map is restricted to the face — a different background
    # would otherwise dominate the colour range.
    shape_hw = (target_height, before_norm.shape[1])
    face_mask = build_region_mask(
        lm_before_crop.pixels_xy, FACE_OVAL_INDICES, shape_hw,
    ) | build_region_mask(
        lm_after_crop.pixels_xy, FACE_OVAL_INDICES, shape_hw,
    )

    views = AestheticsViews(
        before=build_face_view(before_norm, wrinkles_before),
        after=build_face_view(after_norm, wrinkles_after),
        change_map=build_change_map(before_norm, after_norm, mask=face_mask),
    )

    return AestheticsAnalysis(
        views=views,
        region_deltas=deltas,
        protocol_checks=checks,
        pose_before=pose_before,
        pose_after=pose_after,
        alignment=transform,
        target_height=target_height,
    )


async def compare(
    photo_before: UploadFile,
    photo_after: UploadFile,
    *,
    date_before: str | None = None,
    date_after: str | None = None,
    patient_id: str | None = None,
    before_name: str | None = None,
    after_name: str | None = None,
    target_height: int | None = None,
    wrinkle_threshold: float | None = None,
    settings: Settings,
) -> bytes:
    """Compare a "before" and an "after" Botox photo; return the PDF bytes.

    ``target_height`` and ``wrinkle_threshold`` override the service
    defaults (:mod:`app.core.config`) when the caller provides them.
    """
    resolved_height = target_height or settings.aesthetics_target_height_px
    resolved_threshold = (
        wrinkle_threshold
        if wrinkle_threshold is not None
        else settings.aesthetics_wrinkle_threshold_percentile
    )

    with tempfile.TemporaryDirectory(prefix="athenas_aesthetics_") as tmp:
        tmp_dir = Path(tmp)
        rgb_before, rgb_after = await asyncio.gather(
            _read_one(photo_before, tmp_dir),
            _read_one(photo_after, tmp_dir),
        )

        def _work() -> bytes:
            analysis = run_analysis(
                rgb_before,
                rgb_after,
                target_height=resolved_height,
                padding_frac=settings.aesthetics_face_padding_frac,
                wrinkle_threshold=resolved_threshold,
                wrinkle_kernel_length=settings.aesthetics_wrinkle_kernel_length,
                wrinkle_min_line_length=settings.aesthetics_wrinkle_min_line_length,
            )
            return build_aesthetics_pdf(
                analysis,
                patient_id=patient_id,
                date_before=date_before,
                date_after=date_after,
                before_name=before_name,
                after_name=after_name,
            )

        return await asyncio.to_thread(_work)
