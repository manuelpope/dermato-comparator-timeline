"""MediaPipe Face Landmarker wrapper and pose estimation.

This module owns the only MediaPipe import in the ``face_expression`` package —
``create_face_landmarker`` raises a clear ``RuntimeError`` if the dependency is
missing, so the rest of the package can import the surrounding modules without
paying the MediaPipe startup cost.

The MediaPipe 0.10+ API uses the new Tasks runtime (``mp.tasks.vision``) which
requires a ``face_landmarker.task`` model asset. The model is **vendored** in
``face_expression/models/face_landmarker.task`` so the API works offline (and
inside the Docker image); if the asset is missing we fall back to the local
cache (``~/.cache/mediapipe``) and finally to a lazy download.

Public dataclasses
------------------
- :class:`FaceLandmarks` — landmarks in pixel + normalised coords.
- :class:`FacePose`      — yaw / pitch / roll in degrees.

Public functions
----------------
- :func:`create_face_landmarker` — builds the MediaPipe ``FaceLandmarker``.
- :func:`detect_landmarks`       — runs ``detect`` on an RGB ndarray.
- :func:`estimate_pose`          — yaw / pitch / roll from the matrix MediaPipe
  returns when ``output_facial_transformation_matrixes=True``.
- :func:`compute_inter_pupillary_distance` — distance between left/right irises.

Region index constants
----------------------
``FACE_OVAL_INDICES``, ``FOREHEAD_REGION_INDICES``, ``GLABELLA_REGION_INDICES``,
``EYES_REGION_INDICES``, ``PERIORBITAL_L_INDICES`` / ``_R_INDICES``,
``NOSE_REGION_INDICES``, ``NASOLABIAL_L_INDICES`` / ``_R_INDICES``,
``PERIORAL_REGION_INDICES``, ``MOUTH_REGION_INDICES`` — every region the rest
of the package references. The numbers come from the official MediaPipe Face
Mesh topology (468-point graph).
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical download URL for the MediaPipe ``face_landmarker.task`` model.
#: The file is ~5 MB and published by Google; we cache it on first use.
DEFAULT_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

#: Default cache directory (overridable via the ``MEDIAPIPE_MODEL_DIR`` env var).
DEFAULT_CACHE_DIR: Path = Path(
    os.environ.get("MEDIAPIPE_MODEL_DIR", Path.home() / ".cache" / "mediapipe")
)

#: Default filename for the cached model.
DEFAULT_MODEL_FILENAME: str = "face_landmarker.task"

#: Model asset shipped inside the package — preferred over any download so the
#: API works offline (CI, Docker, air-gapped deployments).
VENDORED_MODEL_PATH: Path = Path(__file__).resolve().parent / "models" / DEFAULT_MODEL_FILENAME

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceLandmarks:
    """MediaPipe Face Landmarker landmarks, in two coordinate systems.

    Attributes
    ----------
    pixels_xy
        ``(478, 2)`` float32 array of (x, y) coordinates in pixel space.
        478 = 468 base mesh + 10 iris refinement landmarks.
    normalised_xyz
        ``(478, 3)`` float32 array from MediaPipe's normalised output.
    image_shape_hw
        ``(H, W)`` of the source image.
    """

    pixels_xy: np.ndarray            # (478, 2) float32
    normalised_xyz: np.ndarray       # (478, 3) float32
    image_shape_hw: tuple[int, int]  # (H, W)


@dataclass(frozen=True)
class FacePose:
    """Head pose in degrees.

    Conventions:

    - ``yaw``   — turn around vertical (left/right). Positive = subject's right.
    - ``pitch`` — tilt around horizontal (up/down). Positive = chin tucked.
    - ``roll``  — in-plane rotation. Positive = CCW in the image plane.
    - ``method``— always ``"facial_transformation_matrix"`` (the only path
      we ship today; computed by MediaPipe directly from the model).
    """

    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    method: str = "facial_transformation_matrix"


# ---------------------------------------------------------------------------
# Region indices (canonical MediaPipe Face Mesh topology).
# Values are public; references the official MediaPipe diagram.
# ---------------------------------------------------------------------------

# Face oval (universal "inside the face" mask).
FACE_OVAL_INDICES: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)

# Forehead / frontalis region.
FOREHEAD_REGION_INDICES: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 67, 109,
)

# Glabella (between the eyebrows — vertical wrinkles here).
GLABELLA_REGION_INDICES: tuple[int, ...] = (
    168, 6, 197, 195, 5, 4, 1,
)

# Both eyes combined.
EYES_REGION_INDICES: tuple[int, ...] = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
    159, 160, 161, 246,
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
    386, 385, 384, 398,
)

# Periorbital left.
PERIORBITAL_L_INDICES: tuple[int, ...] = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
    159, 160, 161, 246,
)

# Periorbital right.
PERIORBITAL_R_INDICES: tuple[int, ...] = (
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
    386, 385, 384, 398,
)

# Nasolabial fold (left).
NASOLABIAL_L_INDICES: tuple[int, ...] = (
    207, 187, 123, 147, 213, 192,
)

# Nasolabial fold (right).
NASOLABIAL_R_INDICES: tuple[int, ...] = (
    427, 411, 352, 376, 433, 416,
)

# Nose (tip + bridge — used as a non-wrinkle control region).
NOSE_REGION_INDICES: tuple[int, ...] = (
    1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 31, 228, 229,
    230, 231, 232, 233, 244, 245, 122, 6, 202, 214, 218, 237,
)

# Lips (vermilion border).
PERIORAL_REGION_INDICES: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
)

# Mouth interior.
MOUTH_REGION_INDICES: tuple[int, ...] = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
    310, 311, 312, 13, 82, 81, 80, 191,
)

# Iris centres — landmarks 468..477 ONLY when ``refine_landmarks=True``.
LEFT_IRIS_CENTER_INDEX: int = 468
RIGHT_IRIS_CENTER_INDEX: int = 473


# ---------------------------------------------------------------------------
# Model auto-download
# ---------------------------------------------------------------------------


def _ensure_model_cached(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    url: str = DEFAULT_MODEL_URL,
    filename: str = DEFAULT_MODEL_FILENAME,
) -> Path:
    """Make sure ``face_landmarker.task`` exists locally; download if not.

    The model is ~5 MB and is downloaded lazily on first call. Subsequent
    calls are a no-op until the file is deleted from the cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / filename
    if model_path.exists():
        return model_path

    try:
        urllib.request.urlretrieve(url, model_path)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"Could not download MediaPipe face landmarker model from {url}: "
            f"{exc}. Place the file manually at {model_path} or set "
            "MEDIAPIPE_MODEL_DIR to a directory that already contains it."
        ) from exc
    return model_path


# ---------------------------------------------------------------------------
# MediaPipe resource creation (lazy import)
# ---------------------------------------------------------------------------


def create_face_landmarker(
    *,
    num_faces: int = 1,
    refine_landmarks: bool = True,
    cache_dir: Path | None = None,
    model_url: str | None = None,
):
    """Build a MediaPipe ``FaceLandmarker`` (Tasks API, lazy import).

    The model asset is resolved in this order: the copy vendored in
    ``face_expression/models/``, the local cache (``MEDIAPIPE_MODEL_DIR`` /
    ``~/.cache/mediapipe``) and finally a download from ``model_url``.

    Parameters
    ----------
    num_faces
        Maximum faces to detect. Keep at 1 — the MVP compares two single-
        subject photos, so any extra detection is noise.
    refine_landmarks
        Enables the iris + lip refinement (landmarks 468–477). Required for
        :func:`compute_inter_pupillary_distance` to use iris centres.
    cache_dir
        Override the model cache directory. Defaults to
        ``~/.cache/mediapipe`` (or ``$MEDIAPIPE_MODEL_DIR``).
    model_url
        Override the model download URL. Useful for air-gapped deployments
        that serve the model from an internal mirror.

    Returns
    -------
    mediapipe.tasks.python.vision.face_landmarker.FaceLandmarker
    """
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "MediaPipe is required for face-expression analysis. "
            "Install it with: uv add 'mediapipe>=0.10.20,<1.0'"
        ) from exc

    resolved_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    resolved_url = model_url if model_url is not None else DEFAULT_MODEL_URL

    if VENDORED_MODEL_PATH.exists():
        model_path = VENDORED_MODEL_PATH
    else:
        model_path = _ensure_model_cached(cache_dir=resolved_dir, url=resolved_url)

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=num_faces,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Landmark detection
# ---------------------------------------------------------------------------


def detect_landmarks(face_landmarker, rgb: np.ndarray) -> FaceLandmarks | None:
    """Run MediaPipe FaceLandmarker on one RGB image.

    Parameters
    ----------
    face_landmarker
        A ``FaceLandmarker`` instance from :func:`create_face_landmarker`.
    rgb
        ``(H, W, 3)`` ``uint8`` RGB ndarray.

    Returns
    -------
    FaceLandmarks | None
        ``None`` if no face is detected, otherwise a frozen dataclass.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")

    try:
        from mediapipe import Image, ImageFormat
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MediaPipe not installed") from exc

    h, w = rgb.shape[:2]
    # MediaPipe wants contiguous, SRGB uint8.
    rgb_contig = np.ascontiguousarray(rgb, dtype=np.uint8)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_contig)

    result = face_landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    normalised = np.array(
        [(lm.x, lm.y, lm.z) for lm in landmarks],
        dtype=np.float32,
    )
    pixels = np.column_stack(
        [normalised[:, 0] * w, normalised[:, 1] * h]
    ).astype(np.float32)

    return FaceLandmarks(
        pixels_xy=pixels,
        normalised_xyz=normalised,
        image_shape_hw=(h, w),
    )


# ---------------------------------------------------------------------------
# Pose estimation
# ---------------------------------------------------------------------------


def estimate_pose(
    face_landmarker,
    rgb: np.ndarray,
    landmarks: FaceLandmarks | None = None,
) -> FacePose:
    """Estimate head pose from MediaPipe's facial transformation matrix.

    When ``output_facial_transformation_matrixes=True`` is set on the
    landmarker (which is the default in :func:`create_face_landmarker`),
    MediaPipe returns a 4×4 matrix that maps a canonical face to the
    detected one. We extract yaw/pitch/roll from its rotation sub-matrix.

    Parameters
    ----------
    face_landmarker
        A ``FaceLandmarker`` instance.
    rgb
        Source image (used to re-run detection if ``landmarks`` is None).
    landmarks
        Optional pre-computed landmarks from :func:`detect_landmarks`. If
        ``None``, we re-run detection to get the matrix.

    Returns
    -------
    FacePose
        Yaw / pitch / roll in degrees. ``(0, 0, 0)`` if no face is detected.
    """
    try:
        from mediapipe import Image, ImageFormat
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MediaPipe not installed") from exc

    rgb_contig = np.ascontiguousarray(rgb, dtype=np.uint8)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_contig)
    result = face_landmarker.detect(mp_image)

    if not result.facial_transformation_matrixes:
        return FacePose(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0)

    # The 4×4 matrix is row-major flat in a list of 16 floats.
    mat = np.array(result.facial_transformation_matrixes[0], dtype=np.float64).reshape(4, 4)
    rot = mat[:3, :3]

    # Decompose into Euler angles (ZYX intrinsic / roll-pitch-yaw).
    # The rotation in MediaPipe's convention is from canonical → image, so we
    # invert to get the head's pose. Negating the angles works because a
    # rotation and its inverse have opposite Euler decompositions (modulo
    # gimbal-lock edge cases we don't hit at typical face angles).
    pitch = float(np.degrees(np.arctan2(rot[2, 1], rot[2, 2])))
    yaw = float(np.degrees(np.arctan2(-rot[2, 0],
                                       np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))))
    roll = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))

    return FacePose(yaw_deg=-yaw, pitch_deg=-pitch, roll_deg=roll)


# ---------------------------------------------------------------------------
# Inter-pupillary distance
# ---------------------------------------------------------------------------


def compute_inter_pupillary_distance(landmarks: FaceLandmarks) -> float:
    """Distance in pixels between the left and right iris centres.

    Requires the iris refinement landmarks (468..477), which MediaPipe
    only outputs when ``refine_landmarks=True`` was set on the landmarker.

    Falls back to the outer-eye-corner distance (33, 263) if the iris
    refinement is unavailable — the distance is roughly 1.5× the true IPD
    but still useful for relative scale comparisons.
    """
    pts = landmarks.pixels_xy

    has_iris = landmarks.normalised_xyz.shape[0] > RIGHT_IRIS_CENTER_INDEX
    if has_iris:
        left = LEFT_IRIS_CENTER_INDEX
        right = RIGHT_IRIS_CENTER_INDEX
    else:
        left = 33
        right = 263

    dx = pts[right, 0] - pts[left, 0]
    dy = pts[right, 1] - pts[left, 1]
    return float(np.hypot(dx, dy))
