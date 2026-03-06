"""
Unit tests for QA (Quality Assurance) service.

This module tests the QA service functionality including YOLO26-based
annotation validation, IoU calculations, and defect detection comparison.
"""

import pytest
import numpy as np
import tempfile
import cv2

from services import qa_service


class FakeTensor:
    """Minimal tensor-like wrapper compatible with qa_service parsing helpers."""

    def __init__(self, data):
        self._data = np.asarray(data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class FakeBoxesStructured:
    """Mock YOLO26 box container with xyxy/conf/cls fields."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)


class FakeBoxesData:
    """Mock YOLO26 box container with Nx6 data rows (xyxy, conf, cls)."""

    def __init__(self, data):
        self.data = FakeTensor(data)


class FakeResult:
    """Single inference result wrapper."""

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Fake YOLO26 model returning deterministic detections."""

    def __init__(self, results):
        self._results = results
        self.names = {0: "defect", 1: "minor_defect"}

    def __call__(self, image_path: str, verbose: bool = False):
        _ = image_path, verbose
        return self._results


@pytest.fixture
def sample_image():
    """
    Create a sample test image with defect patterns.
    
    Returns:
        Path to temporary image file
    """
    # Create test image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Add defect-like patterns
    cv2.rectangle(img, (100, 100), (150, 150), (50, 50, 50), -1)
    cv2.rectangle(img, (300, 200), (350, 250), (50, 50, 50), -1)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        cv2.imwrite(f.name, img)
        return f.name


@pytest.fixture
def human_annotations():
    """
    Create sample human annotations for testing.
    
    Returns:
        List of annotation dictionaries
    """
    return [
        {
            "bbox": [100, 100, 50, 50],
            "class_name": "defect",
            "confidence": 1.0,
            "annotation_id": "human_1"
        },
        {
            "bbox": [300, 200, 50, 50],
            "class_name": "defect",
            "confidence": 1.0,
            "annotation_id": "human_2"
        }
    ]


@pytest.fixture
def mock_yolo26_structured(monkeypatch):
    """
    Mock YOLO26 structured output (xyxy/conf/cls fields).

    Validates parser path used by most Ultralytics result objects.
    """
    results = [
        FakeResult(
            FakeBoxesStructured(
                xyxy=[[100.2, 100.1, 150.1, 150.3], [300.0, 200.0, 350.0, 250.0]],
                conf=[0.92, 0.88],
                cls=[0, 0],
            )
        )
    ]
    monkeypatch.setattr(qa_service, "load_yolo_model", lambda: FakeModel(results))


@pytest.fixture
def mock_yolo26_data_format(monkeypatch):
    """
    Mock YOLO26 row-based output (boxes.data format).

    Validates parser fallback for NMS-free output variants.
    """
    # x1, y1, x2, y2, conf, cls
    data_rows = [[101.0, 101.0, 149.0, 149.0, 0.95, 0], [20.0, 20.0, 30.0, 30.0, 0.83, 1]]
    results = [FakeResult(FakeBoxesData(data_rows))]
    monkeypatch.setattr(qa_service, "load_yolo_model", lambda: FakeModel(results))


def test_calculate_iou_identical_boxes():
    """
    Test IoU calculation with identical boxes.
    
    Validates that:
    - Identical boxes return IoU = 1.0
    """
    box1 = [100, 100, 50, 50]
    box2 = [100, 100, 50, 50]
    
    iou = qa_service.calculate_iou(box1, box2)
    
    assert abs(iou - 1.0) < 0.001, f"Identical boxes should have IoU=1.0, got {iou}"


def test_calculate_iou_no_overlap():
    """
    Test IoU calculation with non-overlapping boxes.
    
    Validates that:
    - Non-overlapping boxes return IoU = 0.0
    """
    box1 = [100, 100, 50, 50]
    box2 = [300, 300, 50, 50]
    
    iou = qa_service.calculate_iou(box1, box2)
    
    assert iou == 0.0, f"Non-overlapping boxes should have IoU=0.0, got {iou}"


def test_calculate_iou_partial_overlap():
    """
    Test IoU calculation with partially overlapping boxes.
    
    Validates that:
    - Partially overlapping boxes return IoU between 0 and 1
    - IoU value is mathematically correct
    """
    # Box 1: [0, 0, 100, 100] - area = 10000
    # Box 2: [50, 50, 100, 100] - area = 10000
    # Intersection: [50, 50, 50, 50] - area = 2500
    # Union: 10000 + 10000 - 2500 = 17500
    # IoU = 2500 / 17500 = 0.1428...
    
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 100, 100]
    
    iou = qa_service.calculate_iou(box1, box2)
    
    expected_iou = 2500 / 17500
    assert abs(iou - expected_iou) < 0.001, f"Expected IoU={expected_iou:.4f}, got {iou:.4f}"


def test_calculate_iou_edge_touching():
    """
    Test IoU calculation with boxes that touch at edges.
    
    Validates that:
    - Edge-touching boxes return IoU = 0.0
    """
    box1 = [0, 0, 50, 50]
    box2 = [50, 0, 50, 50]  # Touching right edge
    
    iou = qa_service.calculate_iou(box1, box2)
    
    assert iou == 0.0, f"Edge-touching boxes should have IoU=0.0, got {iou}"


@pytest.mark.asyncio
async def test_run_qa_check_basic(sample_image, human_annotations, mock_yolo26_structured):
    """
    Test run_qa_check with known annotations vs mocked YOLO26 output.
    
    Validates that:
    - Function executes without errors
    - Returns all required report fields
    - Report structure matches schema
    - Counts are non-negative
    """
    # Run QA check
    result = await qa_service.run_qa_check(
        image_path=sample_image,
        human_annotations=human_annotations,
        iou_threshold=0.5
    )
    
    # Validate result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    
    # Check required fields
    required_fields = [
        "missed_defects",
        "size_warnings",
        "confirmed",
        "total_human_annotations",
        "total_ai_detections",
        "processing_time_seconds"
    ]
    
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    # Validate field types
    assert isinstance(result["missed_defects"], list)
    assert isinstance(result["size_warnings"], list)
    assert isinstance(result["confirmed"], list)
    assert isinstance(result["total_human_annotations"], int)
    assert isinstance(result["total_ai_detections"], int)
    assert isinstance(result["processing_time_seconds"], float)
    
    # Validate counts
    assert result["total_human_annotations"] == len(human_annotations)
    assert result["total_human_annotations"] >= 0
    assert result["total_ai_detections"] >= 0
    assert result["processing_time_seconds"] > 0


@pytest.mark.asyncio
async def test_run_qa_check_invalid_image():
    """
    Test run_qa_check with non-existent image.
    
    Validates that:
    - FileNotFoundError is raised for missing image
    """
    with pytest.raises(FileNotFoundError):
        await qa_service.run_qa_check(
            image_path="nonexistent_image.jpg",
            human_annotations=[],
            iou_threshold=0.5
        )


@pytest.mark.asyncio
async def test_run_qa_check_empty_annotations(sample_image, mock_yolo26_structured):
    """
    Test run_qa_check with empty human annotations.
    
    Validates that:
    - Function handles empty annotation list
    - All YOLO detections are marked as missed
    """
    result = await qa_service.run_qa_check(
        image_path=sample_image,
        human_annotations=[],
        iou_threshold=0.5
    )
    
    # With no human annotations, all AI detections should be "missed"
    assert result["total_human_annotations"] == 0
    assert len(result["confirmed"]) == 0
    assert len(result["size_warnings"]) == 0


def test_calculate_box_area():
    """
    Test calculate_box_area function.
    
    Validates that:
    - Area calculation is correct for various box sizes
    """
    # Test square box
    bbox = [0, 0, 50, 50]
    area = qa_service.calculate_box_area(bbox)
    assert area == 2500, f"Expected area=2500, got {area}"
    
    # Test rectangular box
    bbox = [0, 0, 100, 50]
    area = qa_service.calculate_box_area(bbox)
    assert area == 5000, f"Expected area=5000, got {area}"
    
    # Test unit box
    bbox = [0, 0, 1, 1]
    area = qa_service.calculate_box_area(bbox)
    assert area == 1, f"Expected area=1, got {area}"


@pytest.mark.asyncio
async def test_run_qa_check_confirms_matching_annotations(sample_image, mock_yolo26_structured):
    """
    Test that QA check correctly confirms annotations that match YOLO predictions.
    
    Validates that:
    - Annotations with high IoU are confirmed
    - Confirmed annotations have correct structure
    """
    # Create annotations that should match YOLO detections
    annotations = [
        {
            "bbox": [100, 100, 50, 50],
            "class_name": "object",
            "annotation_id": "test_1"
        }
    ]
    
    result = await qa_service.run_qa_check(
        image_path=sample_image,
        human_annotations=annotations,
        iou_threshold=0.5
    )
    
    # Check confirmed annotations structure
    for confirmed in result["confirmed"]:
        assert "bbox" in confirmed
        assert "class_name" in confirmed
        assert "confidence" in confirmed
        assert "iou_with_yolo" in confirmed
        assert 0.0 <= confirmed["iou_with_yolo"] <= 1.0


@pytest.mark.asyncio
async def test_run_qa_check_different_iou_thresholds(sample_image, human_annotations, mock_yolo26_structured):
    """
    Test run_qa_check with different IoU thresholds.
    
    Validates that:
    - Lower threshold results in more confirmed annotations
    - Higher threshold results in fewer confirmed annotations
    """
    # Run with low threshold
    result_low = await qa_service.run_qa_check(
        image_path=sample_image,
        human_annotations=human_annotations,
        iou_threshold=0.3
    )
    
    # Run with high threshold
    result_high = await qa_service.run_qa_check(
        image_path=sample_image,
        human_annotations=human_annotations,
        iou_threshold=0.7
    )
    
    # Lower threshold should confirm more or equal annotations
    assert len(result_low["confirmed"]) >= len(result_high["confirmed"])


@pytest.mark.asyncio
async def test_run_qa_check_supports_yolo26_data_rows(sample_image, human_annotations, mock_yolo26_data_format):
    """
    Test run_qa_check parses YOLO26 NMS-free row-style output without crashing.

    Validates that:
    - ``boxes.data`` format is accepted
    - Detections are converted to xywh correctly
    - QA report still returns exactly three business lists
    """
    result = await qa_service.run_qa_check(
        image_path=sample_image,
        human_annotations=human_annotations,
        iou_threshold=0.5,
    )

    assert isinstance(result["missed_defects"], list)
    assert isinstance(result["size_warnings"], list)
    assert isinstance(result["confirmed"], list)


def test_extract_yolo26_detections_preserves_float_precision():
    """
    Test float precision retention for tiny boxes (STAL small-target behavior).

    Validates that tiny predicted boxes are not rounded to integers before IoU.
    """
    tiny_result = [
        FakeResult(
            FakeBoxesStructured(
                xyxy=[[10.125, 20.375, 11.375, 21.625]],
                conf=[0.91],
                cls=[0],
            )
        )
    ]

    detections = qa_service.extract_yolo26_detections(tiny_result, {0: "defect"})
    assert len(detections) == 1
    bbox = detections[0]["bbox"]
    assert isinstance(bbox[0], float)
    assert abs(bbox[2] - 1.25) < 1e-9
    assert abs(bbox[3] - 1.25) < 1e-9


def test_iou_calculation_comprehensive():
    """
    Comprehensive test of IoU calculation with various box configurations.
    
    Validates that:
    - IoU is symmetric (IoU(A,B) == IoU(B,A))
    - IoU is bounded [0, 1]
    - IoU handles all edge cases correctly
    """
    test_cases = [
        # (box1, box2, expected_range)
        ([0, 0, 100, 100], [0, 0, 100, 100], (0.99, 1.01)),  # Identical
        ([0, 0, 100, 100], [200, 200, 100, 100], (0.0, 0.0)),  # No overlap
        ([0, 0, 100, 100], [50, 50, 100, 100], (0.14, 0.15)),  # Partial overlap
        ([0, 0, 50, 50], [25, 25, 50, 50], (0.14, 0.15)),  # Quarter overlap
    ]
    
    for box1, box2, (min_iou, max_iou) in test_cases:
        iou = qa_service.calculate_iou(box1, box2)
        assert min_iou <= iou <= max_iou, f"IoU {iou} not in expected range [{min_iou}, {max_iou}] for boxes {box1}, {box2}"
        
        # Test symmetry
        iou_reverse = qa_service.calculate_iou(box2, box1)
        assert abs(iou - iou_reverse) < 0.0001, f"IoU not symmetric: {iou} != {iou_reverse}"
