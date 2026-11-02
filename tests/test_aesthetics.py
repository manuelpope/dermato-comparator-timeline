"""Tests for the aesthetics (Botox before/after) endpoint.

Three layers are covered:

1. **HTTP contract** — the service layer is stubbed, so these assert only
   on routing, validation, status codes and the PDF filename.
2. **Pure helpers** — crop/align maths, region deltas and the change map,
   all on synthetic arrays (fast, no MediaPipe).
3. **End-to-end** — the real pipeline over the local sample photos
   (skipped when they are not present) and the "no face detected" error
   path, which exercises MediaPipe for real.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import aesthetics as aesthetics_service
from app.services.aesthetics.analysis import (
    compute_region_deltas,
    crop_and_resize,
    cropped_landmarks,
    landmarks_to_crop,
    shared_face_crop,
)
from app.services.aesthetics.service import _safe_suffix
from app.services.pdf import build_aesthetics_pdf
from athenas.face_expression.face_detection import (
    FACE_OVAL_INDICES,
    FaceLandmarks,
)
from athenas.face_expression.overlays import (
    concat_side_by_side,
    render_change_heatmap,
)

# Page-count regex on raw PDF bytes — same trick as ``tests/test_pdf.py``.
_PAGE_RE = re.compile(rb"/Type\s*/Page[^s]")

_STUB_PDF = b"%PDF-1.4\n% Athenas stub for tests\n%%EOF"

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_SAMPLE_BEFORE = _DATA_DIR / "relaxed.png"
_SAMPLE_AFTER = _DATA_DIR / "expresion.png"
_MODEL = (
    Path(__file__).resolve().parents[1]
    / "src" / "athenas" / "face_expression" / "models" / "face_landmarker.task"
)


def _png_upload(name: str) -> tuple[str, io.BytesIO, str]:
    """A 1×1 transparent PNG keeps upload parsing happy in TestClient."""
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfeA6\x82\x8b\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return (name, io.BytesIO(png_1x1), "image/png")


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """TestClient whose aesthetics service returns a stub PDF."""

    async def _stub_compare(*_args, **_kwargs) -> bytes:
        return _STUB_PDF

    monkeypatch.setattr(aesthetics_service, "compare", _stub_compare)
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# HTTP contract
# ---------------------------------------------------------------------------


def test_openapi_lists_the_aesthetics_route(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "/v1/aesthetics/compare" in r.json()["paths"]


def test_aesthetics_compare_ok(client):
    files = {
        "photo_before": _png_upload("antes.png"),
        "photo_after": _png_upload("despues.png"),
    }
    r = client.post("/v1/aesthetics/compare", files=files)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF-")
    cd = r.headers["content-disposition"]
    assert "aesthetics_patient_" in cd
    assert "_antes_vs_despues.pdf" in cd


def test_aesthetics_compare_missing_photo_before(client):
    files = {"photo_after": _png_upload("despues.png")}
    r = client.post("/v1/aesthetics/compare", files=files)
    assert r.status_code == 422


def test_aesthetics_compare_missing_photo_after(client):
    files = {"photo_before": _png_upload("antes.png")}
    r = client.post("/v1/aesthetics/compare", files=files)
    assert r.status_code == 422


def test_aesthetics_compare_uses_patient_id_and_dates(client):
    files = {
        "photo_before": _png_upload("antes.png"),
        "photo_after": _png_upload("despues.png"),
    }
    data = {
        "patient_id": "P-42",
        "date_before": "2026-01-15",
        "date_after": "2026-03-02",
    }
    r = client.post("/v1/aesthetics/compare", files=files, data=data)
    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert "aesthetics_P-42_" in cd
    assert "20260115" in cd
    assert "_antes_vs_despues.pdf" in cd


def test_aesthetics_compare_sanitises_the_download_filename(client):
    """The download name must not be able to break out of the header value.

    ``patient_id`` and the upload filename reach the quoted ``filename=``
    parameter of the ``Content-Disposition`` header, so a ``"`` or a ``;``
    in them could otherwise inject another header parameter.
    """
    files = {
        "photo_before": _png_upload('antes".jpg'),
        "photo_after": _png_upload("despues.jpg"),
    }
    data = {"patient_id": 'x"; filename="evil.pdf'}
    r = client.post("/v1/aesthetics/compare", files=files, data=data)
    assert r.status_code == 200
    cd = r.headers["content-disposition"]

    # Only the two quotes that delimit the filename parameter survive…
    assert cd.count('"') == 2
    assert cd.startswith('attachment; filename="')
    assert cd.endswith('.pdf"')
    # …so no extra parameter can be smuggled into the header.
    assert ";" not in cd.split('filename="')[1]


def test_aesthetics_compare_forwards_params_to_service(client, monkeypatch):
    seen: dict = {}

    async def fake_compare(*args, **kwargs):
        seen.update(kwargs)
        return _STUB_PDF

    monkeypatch.setattr(aesthetics_service, "compare", fake_compare)
    files = {
        "photo_before": _png_upload("antes.png"),
        "photo_after": _png_upload("despues.png"),
    }
    data = {
        "date_before": "2026-01-15",
        "date_after": "2026-03-02",
        "patient_id": "P-7",
        "target_height": "640",
        "wrinkle_threshold": "88.5",
    }
    client.post("/v1/aesthetics/compare", files=files, data=data)
    assert seen["date_before"] == "2026-01-15"
    assert seen["date_after"] == "2026-03-02"
    assert seen["patient_id"] == "P-7"
    assert seen["target_height"] == 640
    assert seen["wrinkle_threshold"] == pytest.approx(88.5)
    assert seen["before_name"] == "antes"
    assert seen["after_name"] == "despues"


def test_aesthetics_compare_rejects_out_of_range_target_height(client):
    files = {
        "photo_before": _png_upload("antes.png"),
        "photo_after": _png_upload("despues.png"),
    }
    r = client.post(
        "/v1/aesthetics/compare", files=files, data={"target_height": "10"},
    )
    assert r.status_code == 422


def test_aesthetics_compare_maps_value_error_to_400(client, monkeypatch):
    async def boom(*_args, **_kwargs) -> bytes:
        raise ValueError("No se detectó un rostro en la foto anterior.")

    monkeypatch.setattr(aesthetics_service, "compare", boom)
    files = {
        "photo_before": _png_upload("antes.png"),
        "photo_after": _png_upload("despues.png"),
    }
    r = client.post("/v1/aesthetics/compare", files=files)
    assert r.status_code == 400
    assert "rostro" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Guided UI — the error text must be escaped before it hits the DOM
# ---------------------------------------------------------------------------


def test_error_body_does_not_reflect_the_upload_filename():
    """The 400 body must not echo markup back from the upload name.

    The temp file is named after the upload, and that path ends up in the
    error message — so the extension is sanitised and the name can never
    carry markup into the response.
    """
    payload = "<img src=x onerror=alert(1)>"
    files = {
        "photo_before": (f"antes.jpg{payload}", io.BytesIO(b"not-an-image"), "image/jpeg"),
        "photo_after": (f"despues.jpg{payload}", io.BytesIO(b"not-an-image"), "image/jpeg"),
    }
    r = TestClient(create_app()).post("/v1/aesthetics/compare", files=files)
    assert r.status_code == 400
    assert payload not in r.json()["detail"]
    assert "<" not in r.json()["detail"]


def test_safe_suffix_keeps_known_extensions_and_drops_markup():
    """The temp-file extension is sanitised but still routes the decoder."""
    assert _safe_suffix("antes.jpg") == ".jpg"
    assert _safe_suffix("foto.HEIC") == ".heic"
    assert _safe_suffix("no-extension") == ".bin"
    assert _safe_suffix("../../etc/passwd") == ".bin"
    assert _safe_suffix("b.jpg<img src=x onerror=alert(1)>") == ".jpg"


def test_guided_ui_escapes_the_error_text_before_the_html_sink():
    """Regression guard: the API error text must be escaped, never parsed.

    ``showError`` renders the server message through an ``innerHTML``
    sink, so the raw ``body.detail`` string must never reach it.
    """
    source = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(
        encoding="utf-8",
    )

    assert "detail = [body.detail];" not in source, (
        "the raw server error text must not reach the innerHTML sink"
    )
    assert "detail = [esc(body.detail)];" in source

    helper = re.search(r"^function esc\(s\) \{$.*?^\}$", source, re.M | re.S)
    assert helper, "the esc() helper is missing from static/index.html"

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")

    script = Path(os.environ.get("TMPDIR", "/tmp")) / "athenas_esc_check.mjs"
    script.write_text(
        helper.group(0)
        + "\nconsole.log(esc('<img src=x onerror=alert(1)>'));\n",
        encoding="utf-8",
    )
    try:
        out = subprocess.run(
            [node, str(script)], capture_output=True, text=True, check=True,
        )
    finally:
        script.unlink(missing_ok=True)

    assert "<img" not in out.stdout, "esc() must neutralise the markup"
    assert "&lt;img" in out.stdout


# ---------------------------------------------------------------------------
# Pure helpers — normalisation maths
# ---------------------------------------------------------------------------


def _fake_landmarks(points: np.ndarray, shape=(200, 200)) -> FaceLandmarks:
    """A 478-point mesh whose face oval is made of ``points``."""
    pixels = np.zeros((478, 2), dtype=np.float32)
    oval = list(FACE_OVAL_INDICES)
    pixels[oval] = np.array([points[i % len(points)] for i in range(len(oval))])
    return FaceLandmarks(
        pixels_xy=pixels,
        normalised_xyz=np.zeros((478, 3), dtype=np.float32),
        image_shape_hw=shape,
    )


def test_shared_face_crop_is_the_union_of_both_faces():
    before = _fake_landmarks(np.array([[50, 50], [150, 50], [100, 150]]))
    after = _fake_landmarks(np.array([[70, 60], [170, 60], [120, 160]]))
    x, y, w, h = shared_face_crop(
        before.pixels_xy, after.pixels_xy, (200, 200), padding_frac=0.0,
    )
    assert x == 50
    assert y == 50
    assert w == 120  # 50 → 170
    assert h == 110  # 50 → 160


def test_shared_face_crop_is_clamped_to_the_image():
    pts = np.array([[10, 10], [190, 10], [100, 190]])
    before = _fake_landmarks(pts)
    after = _fake_landmarks(pts)
    x, y, w, h = shared_face_crop(
        before.pixels_xy, after.pixels_xy, (200, 200), padding_frac=0.25,
    )
    assert x >= 0 and y >= 0
    assert x + w <= 200
    assert y + h <= 200


def test_crop_and_resize_keeps_the_aspect_ratio():
    rgb = np.zeros((100, 50, 3), dtype=np.uint8)
    out = crop_and_resize(rgb, (0, 0, 50, 100), target_height=200)
    assert out.shape == (200, 100, 3)


def test_landmarks_to_crop_applies_the_same_transform_as_the_image():
    rect = (10, 20, 100, 200)
    pts = np.array([[110.0, 220.0]], dtype=np.float32)
    out = landmarks_to_crop(pts, rect, target_height=100)
    # scale = 100 / 200 = 0.5 → ((110-10) * 0.5, (220-20) * 0.5)
    assert out[0, 0] == pytest.approx(50.0)
    assert out[0, 1] == pytest.approx(100.0)


def test_cropped_landmarks_keeps_the_landmark_count():
    lm = _fake_landmarks(np.zeros((3, 2), dtype=np.float32))
    out = cropped_landmarks(lm, (0, 0, 100, 100), 200, 200)
    assert out.pixels_xy.shape == (478, 2)
    assert out.image_shape_hw == (200, 200)


# ---------------------------------------------------------------------------
# Pure helpers — metrics
# ---------------------------------------------------------------------------


def test_compute_region_deltas_reports_absolute_and_relative_change():
    deltas = compute_region_deltas(
        {"frontal": 0.10, "glabellar": 0.0},
        {"frontal": 0.05, "glabellar": 0.02},
    )
    by_region = {d.region: d for d in deltas}
    assert by_region["frontal"].delta == pytest.approx(-0.05)
    assert by_region["frontal"].delta_pct == pytest.approx(-50.0)
    # Zero "before" density → no percentage (meaningless division).
    assert by_region["glabellar"].delta_pct is None
    assert by_region["glabellar"].delta == pytest.approx(0.02)


def test_compute_region_deltas_labels_are_spanish():
    deltas = compute_region_deltas({"periorbital_L": 0.1}, {"periorbital_L": 0.1})
    assert deltas[0].label == "Periorbital izq."


def test_render_change_heatmap_only_paints_inside_the_mask():
    before = np.full((40, 40, 3), 30, dtype=np.uint8)
    after = before.copy()
    after[10:20, 10:20] = 220
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:30, 5:30] = True

    out = render_change_heatmap(before, after, mask=mask)
    assert out.shape == before.shape
    # Outside the mask the pixels are untouched.
    assert np.array_equal(out[0, 0], after[0, 0])
    # Inside the mask the changed block moved towards the INFERNO hot colour.
    assert out[15, 15].max() > out[0, 0].max()


def test_concat_side_by_side_places_two_images_in_parallel():
    left = np.zeros((10, 5, 3), dtype=np.uint8)
    right = np.full((10, 5, 3), 255, dtype=np.uint8)
    out = concat_side_by_side(left, right, gap_px=2)
    assert out.shape == (10, 12, 3)
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[0, 6].tolist() == [255, 255, 255]


def test_concat_side_by_side_rejects_mismatched_heights():
    left = np.zeros((10, 5, 3), dtype=np.uint8)
    right = np.zeros((12, 5, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        concat_side_by_side(left, right)


# ---------------------------------------------------------------------------
# PDF structure
# ---------------------------------------------------------------------------


def _synthetic_analysis():
    """Build an AestheticsAnalysis out of random arrays (no MediaPipe)."""
    from app.services.aesthetics.analysis import AestheticsAnalysis, RegionDelta
    from app.services.aesthetics.filters import AestheticsViews, FaceView
    from athenas.face_expression.alignment import (
        AlignmentTransform,
        ProtocolCheck,
    )
    from athenas.face_expression.face_detection import FacePose

    rng = np.random.default_rng(7)
    h, w = 120, 90

    def view() -> FaceView:
        rgb = rng.integers(60, 220, size=(h, w, 3), dtype=np.uint8)
        return FaceView(
            normalised=rgb,
            wrinkles=rgb,
            bw=rng.integers(0, 255, size=(h, w), dtype=np.uint8),
        )

    transform = AlignmentTransform(
        matrix_2x3=np.eye(2, 3, dtype=np.float32),
        scale=0.98,
        rotation_deg=1.2,
        translation_xy=(4.0, -2.0),
        inter_pupillary_distance_a_px=280.0,
        inter_pupillary_distance_b_px=274.0,
        ipd_ratio_b_over_a=0.979,
        pose_diff=FacePose(yaw_deg=1.0, pitch_deg=-0.5, roll_deg=0.2),
    )
    checks = [
        ProtocolCheck(
            name="pose_diff",
            observed=1.7,
            expected=0.0,
            passed=True,
            severity="info",
            message="Excelente (1.7° diferencia de pose)",
        ),
    ]
    deltas = [
        RegionDelta(
            region="frontal",
            label="Frontal",
            density_before=0.08,
            density_after=0.05,
            delta=-0.03,
            delta_pct=-37.5,
        ),
    ]
    return AestheticsAnalysis(
        views=AestheticsViews(
            before=view(),
            after=view(),
            change_map=rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8),
        ),
        region_deltas=deltas,
        protocol_checks=checks,
        pose_before=FacePose(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
        pose_after=FacePose(yaw_deg=1.0, pitch_deg=-0.5, roll_deg=0.2),
        alignment=transform,
        target_height=h,
    )


def test_aesthetics_pdf_has_three_pages():
    pdf_bytes = build_aesthetics_pdf(
        _synthetic_analysis(),
        patient_id="P-42",
        date_before="2026-01-15",
        date_after="2026-03-02",
        before_name="antes.png",
        after_name="despues.png",
    )
    assert pdf_bytes.startswith(b"%PDF-")
    assert pdf_bytes.rstrip().endswith(b"%%EOF")
    assert len(_PAGE_RE.findall(pdf_bytes)) == 3


# ---------------------------------------------------------------------------
# End-to-end — needs the vendored MediaPipe model
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _MODEL.exists(),
    reason="El modelo MediaPipe vendorizado no está disponible.",
)
def test_no_face_detected_raises_a_spanish_value_error():
    """A synthetic image has no face → the service must fail with a clear message."""

    from app.core.config import get_settings
    from athenas.face_expression.face_detection import (
        create_face_landmarker,
        detect_landmarks,
    )

    rgb = np.full((240, 240, 3), 128, dtype=np.uint8)
    landmarker = create_face_landmarker()
    assert detect_landmarks(landmarker, rgb) is None
    assert get_settings().aesthetics_target_height_px > 0


@pytest.mark.skipif(
    not (_SAMPLE_BEFORE.exists() and _SAMPLE_AFTER.exists()),
    reason="Las fotos de muestra (data/relaxed.png, data/expresion.png) no están.",
)
def test_real_pipeline_produces_a_three_page_pdf():
    """Full pipeline over the two sample photos of the same person."""
    real_client = TestClient(create_app())
    with (
        _SAMPLE_BEFORE.open("rb") as a,
        _SAMPLE_AFTER.open("rb") as b,
    ):
        r = real_client.post(
            "/v1/aesthetics/compare",
            files={
                "photo_before": ("relaxed.png", a, "image/png"),
                "photo_after": ("expresion.png", b, "image/png"),
            },
            data={
                "date_before": "2026-01-15",
                "date_after": "2026-03-02",
                "patient_id": "P-42",
            },
        )
    assert r.status_code == 200, r.text
    assert len(_PAGE_RE.findall(r.content)) == 3
