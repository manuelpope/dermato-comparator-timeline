"""FastAPI smoke tests via :class:`fastapi.testclient.TestClient`.

The actual CV / pipeline work is monkey-patched — these only assert that the
HTTP surface (paths, validation, status codes, content types, filenames)
behaves as documented. CV correctness is covered by ``tests/test_smoke.py``
and ``tests/test_filters.py``.
"""

from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import dermatology as derm_service
from app.services import trichology as tri_service

# A 1×1 transparent PNG keeps upload parsing happy in TestClient.
_PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfeA6\x82\x8b\x00\x00\x00\x00IEND\xaeB`\x82"
)

_STUB_PDF = b"%PDF-1.4\n% Athenas stub for tests\n%%EOF"


async def _stub_trichology_compare(*_args, **_kwargs) -> bytes:
    return _STUB_PDF


async def _stub_dermatology_compare(*_args, **_kwargs) -> bytes:
    return _STUB_PDF


@pytest.fixture
def client(monkeypatch) -> TestClient:
    """TestClient whose service-layer functions return a stub PDF."""
    monkeypatch.setattr(tri_service, "compare", _stub_trichology_compare)
    monkeypatch.setattr(derm_service, "compare", _stub_dermatology_compare)
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_openapi_lists_both_routes(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"].keys()
    assert "/v1/trichology/compare" in paths
    assert "/v1/dermatology/compare" in paths


# ---------------------------------------------------------------------------
# Trichology endpoint
# ---------------------------------------------------------------------------


def _png_upload(name: str) -> tuple[str, io.BytesIO, str]:
    return (name, io.BytesIO(_PNG_1X1), "image/png")


def test_trichology_compare_ok(client):
    files = {
        "baseline": _png_upload("a.png"),
        "followup": _png_upload("b.png"),
    }
    r = client.post("/v1/trichology/compare", files=files)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF-")
    # Filename: trichology_<patient>_<YYYYMMDD>_<a_stem>_vs_<b_stem>.pdf
    cd = r.headers["content-disposition"]
    assert "trichology_patient_" in cd
    assert "_a_vs_b.pdf" in cd


def test_trichology_compare_missing_followup(client):
    files = {"baseline": _png_upload("a.png")}
    r = client.post("/v1/trichology/compare", files=files)
    assert r.status_code == 422


def test_trichology_compare_uses_patient_id_and_dates(client):
    files = {
        "baseline": _png_upload("baseline.png"),
        "followup": _png_upload("followup.png"),
    }
    data = {
        "patient_id": "P-42",
        "baseline_date": "2026-01-15",
        "followup_date": "2026-07-11",
    }
    r = client.post("/v1/trichology/compare", files=files, data=data)
    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert "trichology_P-42_" in cd
    assert "_baseline_vs_followup.pdf" in cd


def test_trichology_compare_passes_dates_to_service(client, monkeypatch):
    """Verify the service layer receives the dates as kwargs."""
    seen: dict = {}

    async def fake_compare(*args, **kwargs):
        seen.update(kwargs)
        return _STUB_PDF

    monkeypatch.setattr(tri_service, "compare", fake_compare)
    files = {
        "baseline": _png_upload("a.png"),
        "followup": _png_upload("b.png"),
    }
    data = {"baseline_date": "2026-01-15", "followup_date": "2026-07-11"}
    client.post("/v1/trichology/compare", files=files, data=data)
    assert seen.get("baseline_date") == "2026-01-15"
    assert seen.get("followup_date") == "2026-07-11"


# ---------------------------------------------------------------------------
# Dermatology endpoint — explicit photo_a / photo_b / photo_c fields
# ---------------------------------------------------------------------------


def test_dermatology_compare_with_two_photos(client):
    files = {"photo_a": _png_upload("a.png"), "photo_b": _png_upload("b.png")}
    r = client.post("/v1/dermatology/compare", files=files)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    cd = r.headers["content-disposition"]
    assert "dermatology_patient_" in cd
    assert "_2_lesions.pdf" in cd


def test_dermatology_compare_with_three_photos(client):
    files = {
        "photo_a": _png_upload("a.png"),
        "photo_b": _png_upload("b.png"),
        "photo_c": _png_upload("c.png"),
    }
    r = client.post("/v1/dermatology/compare", files=files)
    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert "_3_lesions.pdf" in cd


def test_dermatology_compare_missing_photo_a(client):
    files = {"photo_b": _png_upload("b.png")}
    r = client.post("/v1/dermatology/compare", files=files)
    assert r.status_code == 422


def test_dermatology_compare_missing_photo_b(client):
    files = {"photo_a": _png_upload("a.png")}
    r = client.post("/v1/dermatology/compare", files=files)
    assert r.status_code == 422


def test_dermatology_compare_with_patient_id_and_dates(client):
    files = {
        "photo_a": _png_upload("a.png"),
        "photo_b": _png_upload("b.png"),
    }
    data = {
        "patient_id": "P-99",
        "date_a": "2026-01-15",
        "date_b": "2026-07-11",
    }
    r = client.post("/v1/dermatology/compare", files=files, data=data)
    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert "dermatology_P-99_" in cd


def test_dermatology_compare_three_photos_with_dates(client):
    files = {
        "photo_a": _png_upload("a.png"),
        "photo_b": _png_upload("b.png"),
        "photo_c": _png_upload("c.png"),
    }
    data = {
        "patient_id": "P-7",
        "date_a": "2025-12-01",
        "date_b": "2026-03-15",
        "date_c": "2026-09-20",
    }
    r = client.post("/v1/dermatology/compare", files=files, data=data)
    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert "dermatology_P-7_" in cd
    assert "_3_lesions.pdf" in cd


def test_dermatology_compare_passes_dates_to_service(client, monkeypatch):
    """Verify date_a/b/c reach the service layer via the photos tuple."""
    seen: dict = {}

    async def fake_compare(*args, **kwargs):
        seen["photos"] = kwargs.get("photos")
        return _STUB_PDF

    monkeypatch.setattr(derm_service, "compare", fake_compare)
    files = {
        "photo_a": _png_upload("a.png"),
        "photo_b": _png_upload("b.png"),
    }
    data = {"date_a": "2026-01-15", "date_b": "2026-07-11"}
    client.post("/v1/dermatology/compare", files=files, data=data)
    photos = seen["photos"]
    assert len(photos) == 2
    assert photos[0][1] == "2026-01-15"
    assert photos[1][1] == "2026-07-11"