"""
HTTP-level tests for detect_router convenience endpoints.

These tests mock all ML service calls so no GPU or model files are required.
They verify the request/response contract the frontend depends on.
"""

import json
import io
import numpy as np
import cv2
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

# Explicit imports so patch() can resolve submodule attributes
import models.database  # noqa: F401
import services.sam_service  # noqa: F401
import services.qa_service  # noqa: F401
import services.patch_service  # noqa: F401
import services.synthetic_service  # noqa: F401


def _make_jpeg(width: int = 64, height: int = 64) -> bytes:
    """Create a minimal valid JPEG for upload tests."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 128
    _, buf = cv2.imencode(".jpg", img)
    return bytes(buf)


def _make_upload_file(data: bytes, filename: str = "test.jpg") -> tuple:
    return (filename, io.BytesIO(data), "image/jpeg")


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def jpeg_bytes():
    return _make_jpeg()


@pytest.fixture
def fake_detections():
    return [
        {"class_name": "defect", "confidence": 0.87, "bbox": [10.0, 20.0, 30.0, 40.0]},
        {"class_name": "bolt", "confidence": 0.92, "bbox": [50.0, 60.0, 20.0, 20.0]},
    ]


@pytest.fixture
async def client():
    """
    AsyncClient with all DB and ML init calls mocked.
    Allows the FastAPI app to start without real infrastructure.
    """
    patches = [
        patch("models.database.init_db", new_callable=AsyncMock),
        patch("models.database.close_db", new_callable=AsyncMock),
        patch("services.sam_service.load_sam_model", return_value=None),
        patch("services.qa_service.load_yolo_model", return_value=None),
        patch("services.patch_service.load_clip_model", return_value=None),
        patch("services.synthetic_service.load_diffusion_pipeline", return_value=None),
    ]
    started = [p.start() for p in patches]
    try:
        from main import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac
    finally:
        for p in patches:
            p.stop()


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/upload-image
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_image_response_structure(client, jpeg_bytes, fake_detections):
    """
    POST /api/upload-image returns the detection envelope the frontend expects.
    Fields: detections (list), total_detections (int), processing_time_seconds (float).
    """
    fake_model = MagicMock()
    fake_model.names = {0: "defect", 1: "bolt"}
    fake_model.return_value = []

    with patch("routers.detect_router.qa_service") as mock_qa, \
         patch("services.storage_service.load_cv2_image", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = (np.zeros((64, 64, 3), dtype=np.uint8), "/tmp/test.jpg")
        mock_qa.load_yolo_model.return_value = fake_model
        mock_qa.extract_yolo26_detections.return_value = fake_detections

        response = await client.post(
            "/api/upload-image",
            files={"file": _make_upload_file(jpeg_bytes)},
        )

    assert response.status_code == 200
    data = response.json()

    assert "detections" in data
    assert "total_detections" in data
    assert "processing_time_seconds" in data
    assert isinstance(data["detections"], list)
    assert isinstance(data["total_detections"], int)
    assert isinstance(data["processing_time_seconds"], float)
    assert data["total_detections"] == len(data["detections"])


@pytest.mark.asyncio
async def test_upload_image_detection_fields(client, jpeg_bytes, fake_detections):
    """Each detection object must have id, class_name, confidence, bbox, source, status."""
    fake_model = MagicMock()
    fake_model.names = {0: "defect"}
    fake_model.return_value = []

    with patch("routers.detect_router.qa_service") as mock_qa, \
         patch("services.storage_service.load_cv2_image", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = (np.zeros((64, 64, 3), dtype=np.uint8), "/tmp/test.jpg")
        mock_qa.load_yolo_model.return_value = fake_model
        mock_qa.extract_yolo26_detections.return_value = [fake_detections[0]]

        response = await client.post(
            "/api/upload-image",
            files={"file": _make_upload_file(jpeg_bytes)},
        )

    data = response.json()
    assert len(data["detections"]) == 1
    det = data["detections"][0]
    assert "id" in det
    assert "class_name" in det
    assert "confidence" in det
    assert "bbox" in det
    assert det["source"] == "AI Detected"
    assert det["status"] == "pending"


@pytest.mark.asyncio
async def test_upload_image_no_file_returns_422(client):
    """Uploading with no file field returns HTTP 422."""
    response = await client.post("/api/upload-image")
    assert response.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/detect/propagate
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def seed_annotations():
    return [
        {"bbox": [10, 20, 30, 40], "class_name": "defect", "confidence": 1.0}
    ]


@pytest.mark.asyncio
async def test_propagate_response_structure(client, jpeg_bytes, seed_annotations):
    """
    POST /api/detect/propagate returns proposed_annotations, total_proposed,
    and processing_time_seconds.
    """
    proposed = [{"bbox": [50, 60, 20, 20], "class_name": "defect", "confidence": 0.78}]

    with patch("routers.detect_router.sam_service") as mock_sam:
        mock_sam.propagate_annotations = AsyncMock(return_value=proposed)

        response = await client.post(
            "/api/detect/propagate",
            files={"file": _make_upload_file(jpeg_bytes)},
            data={
                "seed_annotations": json.dumps(seed_annotations),
                "similarity_threshold": "0.75",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "proposed_annotations" in data
    assert "total_proposed" in data
    assert "processing_time_seconds" in data
    assert data["total_proposed"] == len(data["proposed_annotations"])


@pytest.mark.asyncio
async def test_propagate_invalid_json_returns_400(client, jpeg_bytes):
    """Sending malformed JSON in seed_annotations returns HTTP 400."""
    response = await client.post(
        "/api/detect/propagate",
        files={"file": _make_upload_file(jpeg_bytes)},
        data={"seed_annotations": "not-valid-json"},
    )
    assert response.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/detect/qa-check
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_qa_check_response_structure(client, jpeg_bytes, seed_annotations):
    """
    POST /api/detect/qa-check returns the QA report structure the frontend expects.
    """
    qa_result = {
        "missed_defects": [],
        "size_warnings": [],
        "confirmed": [{"bbox": [10, 20, 30, 40], "class_name": "defect", "confidence": 0.9, "iou_with_yolo": 0.88}],
        "total_human_annotations": 1,
        "total_ai_detections": 1,
        "processing_time_seconds": 0.12,
    }

    with patch("routers.detect_router.qa_service") as mock_qa:
        mock_qa.run_qa_check = AsyncMock(return_value=qa_result)

        response = await client.post(
            "/api/detect/qa-check",
            files={"file": _make_upload_file(jpeg_bytes)},
            data={"annotations": json.dumps(seed_annotations), "iou_threshold": "0.5"},
        )

    assert response.status_code == 200
    data = response.json()
    for field in ("missed_defects", "size_warnings", "confirmed",
                  "total_human_annotations", "total_ai_detections"):
        assert field in data


@pytest.mark.asyncio
async def test_qa_check_invalid_json_returns_400(client, jpeg_bytes):
    """Sending malformed JSON in annotations returns HTTP 400."""
    response = await client.post(
        "/api/detect/qa-check",
        files={"file": _make_upload_file(jpeg_bytes)},
        data={"annotations": "{bad json}"},
    )
    assert response.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/detect/extract-patches
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_extract_patches_response_structure(client, jpeg_bytes, seed_annotations):
    """
    POST /api/detect/extract-patches returns patches, optimal_patch_size, total_patches.
    """
    patch_result = {
        "patches": [{"patch_id": 0, "bbox": [10, 20, 30, 40]}],
        "optimal_patch_size": 200,
        "total_patches": 1,
    }

    with patch("routers.detect_router.patch_service") as mock_patch:
        mock_patch.extract_patches = AsyncMock(return_value=patch_result)

        response = await client.post(
            "/api/detect/extract-patches",
            files={"file": _make_upload_file(jpeg_bytes)},
            data={
                "annotations": json.dumps(seed_annotations),
                "patch_size": "200",
                "padding_factor": "1.5",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "patches" in data
    assert "optimal_patch_size" in data
    assert "total_patches" in data
    assert data["total_patches"] == len(data["patches"])


@pytest.mark.asyncio
async def test_extract_patches_empty_annotations(client, jpeg_bytes):
    """Patch extraction with zero annotations returns a valid (possibly empty) result."""
    patch_result = {"patches": [], "optimal_patch_size": 200, "total_patches": 0}

    with patch("routers.detect_router.patch_service") as mock_patch:
        mock_patch.extract_patches = AsyncMock(return_value=patch_result)

        response = await client.post(
            "/api/detect/extract-patches",
            files={"file": _make_upload_file(jpeg_bytes)},
            data={"annotations": "[]"},
        )

    assert response.status_code == 200
    assert response.json()["total_patches"] == 0
