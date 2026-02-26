"""
Unit tests for SAM label propagation service.

This module tests the SAM service functionality including annotation
propagation, feature extraction, and similarity calculations.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import cv2
from PIL import Image

from services import sam_service


@pytest.fixture
def sample_image():
    """
    Create a sample test image.
    
    Returns:
        Path to temporary image file
    """
    # Create a 640x480 test image with some patterns
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Add some defect-like patterns
    cv2.rectangle(img, (100, 100), (150, 150), (50, 50, 50), -1)
    cv2.rectangle(img, (300, 200), (350, 250), (50, 50, 50), -1)
    cv2.rectangle(img, (500, 300), (550, 350), (50, 50, 50), -1)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        cv2.imwrite(f.name, img)
        return f.name


@pytest.fixture
def seed_annotations():
    """
    Create sample seed annotations for testing.
    
    Returns:
        List of annotation dictionaries
    """
    return [
        {
            "bbox": [100, 100, 50, 50],
            "class_name": "defect",
            "confidence": 1.0,
            "annotation_id": "seed_1"
        },
        {
            "bbox": [300, 200, 50, 50],
            "class_name": "defect",
            "confidence": 1.0,
            "annotation_id": "seed_2"
        }
    ]


@pytest.mark.asyncio
async def test_propagate_annotations_with_sample_image(sample_image, seed_annotations):
    """
    Test propagate_annotations with a sample image and seed boxes.
    
    Validates that:
    - Function executes without errors
    - Returns a list of proposed annotations
    - Proposed annotations have required fields
    - Similarity scores are within valid range
    """
    # Run propagation
    result = await sam_service.propagate_annotations(
        image_path=sample_image,
        seed_annotations=seed_annotations,
        similarity_threshold=0.70
    )
    
    # Validate result structure
    assert isinstance(result, list), "Result should be a list"
    
    # Check each proposed annotation
    for annotation in result:
        assert "bbox" in annotation, "Annotation must have 'bbox'"
        assert "class_name" in annotation, "Annotation must have 'class_name'"
        assert "confidence" in annotation, "Annotation must have 'confidence'"
        
        # Validate bbox format
        bbox = annotation["bbox"]
        assert len(bbox) == 4, "Bbox should have 4 values [x, y, w, h]"
        assert all(v >= 0 for v in bbox), "Bbox values should be non-negative"
        
        # Validate confidence score
        confidence = annotation["confidence"]
        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} not in [0, 1]"


@pytest.mark.asyncio
async def test_propagate_annotations_invalid_image():
    """
    Test propagate_annotations with non-existent image.
    
    Validates that:
    - FileNotFoundError is raised for missing image
    """
    with pytest.raises(FileNotFoundError):
        await sam_service.propagate_annotations(
            image_path="nonexistent_image.jpg",
            seed_annotations=[{"bbox": [0, 0, 10, 10], "class_name": "defect"}],
            similarity_threshold=0.75
        )


@pytest.mark.asyncio
async def test_propagate_annotations_insufficient_seeds(sample_image):
    """
    Test propagate_annotations with insufficient seed annotations.
    
    Validates that:
    - ValueError is raised when fewer than 2 seeds provided
    """
    with pytest.raises(ValueError):
        await sam_service.propagate_annotations(
            image_path=sample_image,
            seed_annotations=[{"bbox": [0, 0, 10, 10], "class_name": "defect"}],
            similarity_threshold=0.75
        )


def test_calculate_bbox_similarity():
    """
    Test calculate_bbox_similarity function.
    
    Validates that:
    - Identical boxes have similarity close to 1.0
    - Non-overlapping boxes have similarity close to 0.0
    - Partially overlapping boxes have intermediate similarity
    """
    # Test identical boxes
    bbox1 = [100, 100, 50, 50]
    bbox2 = [100, 100, 50, 50]
    similarity = sam_service.calculate_bbox_similarity(bbox1, bbox2)
    assert similarity > 0.95, f"Identical boxes should have high similarity, got {similarity}"
    
    # Test non-overlapping boxes
    bbox1 = [100, 100, 50, 50]
    bbox2 = [300, 300, 50, 50]
    similarity = sam_service.calculate_bbox_similarity(bbox1, bbox2)
    assert similarity < 0.5, f"Non-overlapping boxes should have low similarity, got {similarity}"
    
    # Test partially overlapping boxes
    bbox1 = [100, 100, 50, 50]
    bbox2 = [120, 120, 50, 50]
    similarity = sam_service.calculate_bbox_similarity(bbox1, bbox2)
    assert 0.2 < similarity < 0.8, f"Partially overlapping boxes should have medium similarity, got {similarity}"


def test_extract_features_from_bbox(sample_image):
    """
    Test extract_features_from_bbox function.
    
    Validates that:
    - Function returns a feature vector
    - Feature vector has correct dimensionality (128)
    - Features are numeric and finite
    """
    # Load image
    image = cv2.imread(sample_image)
    
    # Extract features from a bbox
    bbox = [100, 100, 50, 50]
    features = sam_service.extract_features_from_bbox(image, bbox)
    
    # Validate feature vector
    assert isinstance(features, np.ndarray), "Features should be numpy array"
    assert features.shape == (128,), f"Feature vector should be 128-dim, got {features.shape}"
    assert np.all(np.isfinite(features)), "All features should be finite"


@pytest.mark.asyncio
async def test_propagate_annotations_fallback(sample_image, seed_annotations):
    """
    Test fallback propagation method when SAM is not available.
    
    Validates that:
    - Fallback method executes without errors
    - Returns valid annotations
    - Annotations have required structure
    """
    # Load image
    image = cv2.imread(sample_image)
    
    # Run fallback method
    result = await sam_service.propagate_annotations_fallback(
        image=image,
        seed_annotations=seed_annotations,
        similarity_threshold=0.70
    )
    
    # Validate result
    assert isinstance(result, list), "Result should be a list"
    
    for annotation in result:
        assert "bbox" in annotation
        assert "class_name" in annotation
        assert "confidence" in annotation
        assert len(annotation["bbox"]) == 4


def test_feature_extraction_edge_cases(sample_image):
    """
    Test feature extraction with edge cases.
    
    Validates that:
    - Function handles bboxes at image boundaries
    - Function handles very small bboxes
    - Function handles invalid bboxes gracefully
    """
    image = cv2.imread(sample_image)
    img_height, img_width = image.shape[:2]
    
    # Test bbox at image boundary
    bbox = [0, 0, 50, 50]
    features = sam_service.extract_features_from_bbox(image, bbox)
    assert features.shape == (128,)
    
    # Test bbox at bottom-right corner
    bbox = [img_width - 50, img_height - 50, 50, 50]
    features = sam_service.extract_features_from_bbox(image, bbox)
    assert features.shape == (128,)
    
    # Test very small bbox
    bbox = [100, 100, 5, 5]
    features = sam_service.extract_features_from_bbox(image, bbox)
    assert features.shape == (128,)
    
    # Test bbox extending beyond image
    bbox = [img_width - 10, img_height - 10, 50, 50]
    features = sam_service.extract_features_from_bbox(image, bbox)
    assert features.shape == (128,)
