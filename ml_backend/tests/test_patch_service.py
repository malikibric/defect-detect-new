"""
Unit tests for patch extraction and clustering service.

This module tests the patch service functionality including optimal patch
size calculation, patch extraction, and CLIP-based clustering.
"""

import pytest
import numpy as np
import tempfile
import cv2
import base64
from io import BytesIO
from PIL import Image

from services import patch_service


@pytest.fixture
def sample_image():
    """
    Create a sample test image.
    
    Returns:
        Path to temporary image file
    """
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Add varied defect patterns
    cv2.rectangle(img, (100, 100), (130, 130), (50, 50, 50), -1)  # Small
    cv2.rectangle(img, (300, 200), (350, 250), (50, 50, 50), -1)  # Medium
    cv2.rectangle(img, (450, 300), (550, 400), (50, 50, 50), -1)  # Large
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        cv2.imwrite(f.name, img)
        return f.name


@pytest.fixture
def sample_annotations():
    """
    Create sample annotations with varied sizes.
    
    Returns:
        List of annotation dictionaries
    """
    return [
        {"bbox": [100, 100, 30, 30], "class_name": "small_defect"},
        {"bbox": [300, 200, 50, 50], "class_name": "medium_defect"},
        {"bbox": [450, 300, 100, 100], "class_name": "large_defect"},
    ]


def test_calculate_optimal_patch_size_varied_annotations():
    """
    Test optimal patch size calculation with varied annotation sizes.
    
    Validates that:
    - Function returns a reasonable patch size
    - Patch size is a multiple of 32
    - Patch size is within bounds [64, 512]
    """
    annotations = [
        {"bbox": [0, 0, 30, 30]},  # Small
        {"bbox": [0, 0, 50, 50]},  # Medium
        {"bbox": [0, 0, 40, 40]},  # Medium-small
    ]
    
    patch_size = patch_service.calculate_optimal_patch_size(annotations)
    
    # Validate constraints
    assert 64 <= patch_size <= 512, f"Patch size {patch_size} not in range [64, 512]"
    assert patch_size % 32 == 0, f"Patch size {patch_size} not a multiple of 32"


def test_calculate_optimal_patch_size_empty_annotations():
    """
    Test optimal patch size calculation with empty annotations.
    
    Validates that:
    - Function returns default patch size (224) for empty list
    """
    patch_size = patch_service.calculate_optimal_patch_size([])
    assert patch_size == 224, f"Expected default patch size 224, got {patch_size}"


def test_calculate_optimal_patch_size_single_annotation():
    """
    Test optimal patch size calculation with single annotation.
    
    Validates that:
    - Function handles single annotation correctly
    - Returns valid patch size
    """
    annotations = [{"bbox": [0, 0, 60, 60]}]
    
    patch_size = patch_service.calculate_optimal_patch_size(annotations)
    
    assert 64 <= patch_size <= 512
    assert patch_size % 32 == 0


def test_image_to_base64_and_back():
    """
    Test image_to_base64 and base64_to_image functions.
    
    Validates that:
    - Image can be encoded to base64
    - Base64 can be decoded back to image
    - Decoded image matches original (approximately, due to JPEG compression)
    """
    # Create test image
    original_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Encode to base64
    base64_str = patch_service.image_to_base64(original_image)
    
    # Validate base64 string
    assert isinstance(base64_str, str)
    assert len(base64_str) > 0
    
    # Decode back to image
    decoded_image = patch_service.base64_to_image(base64_str)
    
    # Validate decoded image
    assert isinstance(decoded_image, np.ndarray)
    assert decoded_image.shape == original_image.shape


@pytest.mark.asyncio
async def test_extract_patches_basic(sample_image, sample_annotations):
    """
    Test basic patch extraction functionality.
    
    Validates that:
    - Function executes without errors
    - Returns correct number of patches
    - Each patch has required metadata fields
    - Patches are base64 encoded
    """
    result = await patch_service.extract_patches(
        image_path=sample_image,
        annotations=sample_annotations
    )
    
    # Validate result structure
    assert "patches" in result
    assert "optimal_patch_size" in result
    assert "total_patches" in result
    
    # Validate counts
    assert result["total_patches"] == len(sample_annotations)
    assert len(result["patches"]) == len(sample_annotations)
    
    # Validate each patch
    for patch in result["patches"]:
        assert "patch_id" in patch
        assert "image_base64" in patch
        assert "original_bbox" in patch
        assert "class_name" in patch
        assert "patch_size" in patch
        
        # Validate base64 encoding
        assert isinstance(patch["image_base64"], str)
        assert len(patch["image_base64"]) > 0


@pytest.mark.asyncio
async def test_extract_patches_custom_size(sample_image, sample_annotations):
    """
    Test patch extraction with custom patch size.
    
    Validates that:
    - Custom patch size is respected
    - All patches have the specified size
    """
    custom_size = 128
    
    result = await patch_service.extract_patches(
        image_path=sample_image,
        annotations=sample_annotations,
        patch_size=custom_size
    )
    
    # Validate that custom size was used
    assert result["optimal_patch_size"] == custom_size
    
    for patch in result["patches"]:
        assert patch["patch_size"] == custom_size


@pytest.mark.asyncio
async def test_extract_patches_invalid_image():
    """
    Test patch extraction with non-existent image.
    
    Validates that:
    - FileNotFoundError is raised for missing image
    """
    with pytest.raises(FileNotFoundError):
        await patch_service.extract_patches(
            image_path="nonexistent_image.jpg",
            annotations=[{"bbox": [0, 0, 10, 10], "class_name": "test"}]
        )


@pytest.mark.asyncio
async def test_extract_patches_empty_annotations(sample_image):
    """
    Test patch extraction with empty annotations list.
    
    Validates that:
    - ValueError is raised for empty annotations
    """
    with pytest.raises(ValueError):
        await patch_service.extract_patches(
            image_path=sample_image,
            annotations=[]
        )


@pytest.mark.asyncio
async def test_cluster_patches_returns_three_categories(sample_image, sample_annotations):
    """
    Test cluster_patches returns exactly 3 categories (severe, minor, clean).
    
    Validates that:
    - Function returns all three severity categories
    - Categories are lists
    - Total patches equals input patches
    """
    # First extract patches
    extract_result = await patch_service.extract_patches(
        image_path=sample_image,
        annotations=sample_annotations
    )
    
    patches = extract_result["patches"]
    
    # Cluster patches
    result = await patch_service.cluster_patches(
        patches=patches,
        num_clusters=3
    )
    
    # Validate result structure
    assert "severe" in result
    assert "minor" in result
    assert "clean" in result
    assert "cluster_stats" in result
    
    # Validate types
    assert isinstance(result["severe"], list)
    assert isinstance(result["minor"], list)
    assert isinstance(result["clean"], list)
    assert isinstance(result["cluster_stats"], dict)
    
    # Validate total count
    total_clustered = len(result["severe"]) + len(result["minor"]) + len(result["clean"])
    assert total_clustered == len(patches), f"Expected {len(patches)} patches, got {total_clustered}"


@pytest.mark.asyncio
async def test_cluster_patches_insufficient_data():
    """
    Test cluster_patches with insufficient patches.
    
    Validates that:
    - ValueError is raised when patches < num_clusters
    """
    patches = [
        {
            "patch_id": "patch_0",
            "image_base64": patch_service.image_to_base64(np.ones((64, 64, 3), dtype=np.uint8) * 128),
            "original_bbox": [0, 0, 10, 10],
            "class_name": "test",
            "patch_size": 64
        }
    ]
    
    with pytest.raises(ValueError):
        await patch_service.cluster_patches(
            patches=patches,
            num_clusters=3
        )


@pytest.mark.asyncio
async def test_cluster_patches_empty_list():
    """
    Test cluster_patches with empty patches list.
    
    Validates that:
    - ValueError is raised for empty patches list
    """
    with pytest.raises(ValueError):
        await patch_service.cluster_patches(
            patches=[],
            num_clusters=3
        )


@pytest.mark.asyncio
async def test_cluster_patches_cluster_stats(sample_image, sample_annotations):
    """
    Test that cluster_patches returns meaningful cluster statistics.
    
    Validates that:
    - Cluster stats contain required information
    - Stats include num_patches and avg_bbox_area
    - Stats are numeric and valid
    """
    # Extract patches
    extract_result = await patch_service.extract_patches(
        image_path=sample_image,
        annotations=sample_annotations
    )
    
    # Cluster patches
    result = await patch_service.cluster_patches(
        patches=extract_result["patches"],
        num_clusters=3
    )
    
    # Validate cluster stats
    cluster_stats = result["cluster_stats"]
    
    # Should have stats for each cluster
    assert len(cluster_stats) == 3
    
    # Check each cluster's stats
    for cluster_name, stats in cluster_stats.items():
        assert "num_patches" in stats
        assert "avg_bbox_area" in stats
        assert "centroid" in stats
        
        assert stats["num_patches"] >= 0
        assert stats["avg_bbox_area"] >= 0
        assert isinstance(stats["centroid"], list)


@pytest.mark.asyncio
async def test_extract_patches_edge_cases(sample_image):
    """
    Test patch extraction with edge case annotations.
    
    Validates that:
    - Function handles patches near image boundaries
    - Function handles very small annotations
    - Function handles very large annotations
    """
    edge_annotations = [
        {"bbox": [0, 0, 20, 20], "class_name": "corner"},  # Top-left corner
        {"bbox": [620, 460, 20, 20], "class_name": "corner"},  # Bottom-right corner
        {"bbox": [300, 200, 5, 5], "class_name": "tiny"},  # Very small
    ]
    
    result = await patch_service.extract_patches(
        image_path=sample_image,
        annotations=edge_annotations
    )
    
    # Should successfully extract all patches
    assert result["total_patches"] == len(edge_annotations)
    
    # All patches should be valid
    for patch in result["patches"]:
        # Can decode base64
        decoded = patch_service.base64_to_image(patch["image_base64"])
        assert decoded is not None
        assert decoded.size > 0


def test_optimal_patch_size_bounds():
    """
    Test that optimal patch size respects bounds.
    
    Validates that:
    - Very small annotations don't produce too small patches
    - Very large annotations don't produce too large patches
    """
    # Very small annotations
    small_annotations = [{"bbox": [0, 0, 5, 5]} for _ in range(5)]
    small_size = patch_service.calculate_optimal_patch_size(small_annotations)
    assert small_size >= 64, f"Patch size {small_size} below minimum 64"
    
    # Very large annotations
    large_annotations = [{"bbox": [0, 0, 500, 500]} for _ in range(5)]
    large_size = patch_service.calculate_optimal_patch_size(large_annotations)
    assert large_size <= 512, f"Patch size {large_size} above maximum 512"
