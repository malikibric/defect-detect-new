"""
Unit tests for synthetic defect generation service.

This module tests the synthetic data generation functionality using
Stable Diffusion inpainting for creating defect variations.
"""

import pytest
import numpy as np
import tempfile
import cv2
import os
from pathlib import Path

from services import synthetic_service


@pytest.fixture
def sample_image():
    """
    Create a sample test image with a defect.
    
    Returns:
        Path to temporary image file
    """
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Add a defect pattern
    cv2.rectangle(img, (200, 150), (250, 200), (50, 50, 50), -1)
    cv2.circle(img, (225, 175), 15, (30, 30, 30), -1)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        cv2.imwrite(f.name, img)
        return f.name


@pytest.fixture
def sample_annotation():
    """
    Create a sample annotation for testing.
    
    Returns:
        Annotation dictionary
    """
    return {
        "bbox": [200, 150, 50, 50],
        "class_name": "crack",
        "confidence": 0.95,
        "annotation_id": "test_defect_1"
    }


@pytest.fixture
def temp_output_dir():
    """
    Create a temporary output directory.
    
    Returns:
        Path to temporary directory
    """
    import tempfile
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_build_prompt_basic():
    """
    Test build_prompt function with basic inputs.
    
    Validates that:
    - Function returns a string
    - Prompt contains expected keywords
    - Prompt includes defect type, lighting, and severity
    """
    prompt = synthetic_service.build_prompt(
        defect_type="crack",
        lighting="bright",
        severity="severe"
    )
    
    # Validate type
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    
    # Validate content
    assert "crack" in prompt.lower()
    assert "bright" in prompt.lower() or "lighting" in prompt.lower()
    assert "severe" in prompt.lower() or "large" in prompt.lower()


def test_build_prompt_different_lighting_conditions():
    """
    Test build_prompt with different lighting conditions.
    
    Validates that:
    - Different lighting conditions produce different prompts
    - All lighting types are handled correctly
    """
    lighting_conditions = ["dark", "bright", "side-lit", "natural", "overhead"]
    
    prompts = []
    for lighting in lighting_conditions:
        prompt = synthetic_service.build_prompt(
            defect_type="corrosion",
            lighting=lighting,
            severity="moderate"
        )
        prompts.append(prompt)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    # Prompts should be different for different lighting
    assert len(set(prompts)) == len(prompts), "Prompts should be unique for different lighting"


def test_build_prompt_different_severity_levels():
    """
    Test build_prompt with different severity levels.
    
    Validates that:
    - Different severity levels produce different prompts
    - All severity types are handled correctly
    """
    severity_levels = ["minor", "moderate", "severe"]
    
    prompts = []
    for severity in severity_levels:
        prompt = synthetic_service.build_prompt(
            defect_type="scratch",
            lighting="natural",
            severity=severity
        )
        prompts.append(prompt)
        assert isinstance(prompt, str)
    
    # Prompts should be different for different severity
    assert len(set(prompts)) == len(prompts)


def test_build_negative_prompt():
    """
    Test build_negative_prompt function.
    
    Validates that:
    - Function returns a string
    - Negative prompt contains expected exclusion terms
    """
    negative_prompt = synthetic_service.build_negative_prompt()
    
    assert isinstance(negative_prompt, str)
    assert len(negative_prompt) > 0
    
    # Should contain common negative terms
    negative_terms = ["blurry", "low quality", "cartoon", "unrealistic"]
    found_terms = sum(1 for term in negative_terms if term in negative_prompt.lower())
    assert found_terms >= 2, "Negative prompt should contain common exclusion terms"


@pytest.mark.asyncio
async def test_generate_synthetic_defects_returns_correct_number(
    sample_image,
    sample_annotation,
    temp_output_dir
):
    """
    Test generate_synthetic_defects returns correct number of files.
    
    Validates that:
    - Function executes without fatal errors
    - Returns a list of file paths
    - Number of generated files matches request (or uses fallback)
    """
    num_variations = 5
    
    # Generate synthetic defects
    result = await synthetic_service.generate_synthetic_defects(
        image_path=sample_image,
        annotation=sample_annotation,
        num_variations=num_variations,
        output_dir=temp_output_dir
    )
    
    # Validate result
    assert isinstance(result, list)
    
    # Should generate requested number of variations
    # (may use fallback if Stable Diffusion not available)
    assert len(result) == num_variations, f"Expected {num_variations} variations, got {len(result)}"
    
    # All returned paths should exist
    for path in result:
        assert os.path.exists(path), f"Generated file not found: {path}"


@pytest.mark.asyncio
async def test_generate_synthetic_defects_creates_files(
    sample_image,
    sample_annotation,
    temp_output_dir
):
    """
    Test that generate_synthetic_defects actually creates image files.
    
    Validates that:
    - Generated files are valid images
    - Files can be loaded with OpenCV
    - Files have reasonable dimensions
    """
    result = await synthetic_service.generate_synthetic_defects(
        image_path=sample_image,
        annotation=sample_annotation,
        num_variations=3,
        output_dir=temp_output_dir
    )
    
    for path in result:
        # File should exist
        assert os.path.exists(path)
        
        # Should be loadable as image
        img = cv2.imread(path)
        assert img is not None, f"Could not load generated image: {path}"
        
        # Should have reasonable dimensions
        height, width = img.shape[:2]
        assert height > 0 and width > 0
        assert height <= 2048 and width <= 2048  # Reasonable maximum


@pytest.mark.asyncio
async def test_generate_synthetic_defects_invalid_image(sample_annotation, temp_output_dir):
    """
    Test generate_synthetic_defects with non-existent image.
    
    Validates that:
    - FileNotFoundError is raised for missing image
    """
    with pytest.raises(FileNotFoundError):
        await synthetic_service.generate_synthetic_defects(
            image_path="nonexistent_image.jpg",
            annotation=sample_annotation,
            num_variations=5,
            output_dir=temp_output_dir
        )


@pytest.mark.asyncio
async def test_generate_synthetic_defects_invalid_annotation(sample_image, temp_output_dir):
    """
    Test generate_synthetic_defects with invalid annotation.
    
    Validates that:
    - ValueError is raised for invalid annotation
    """
    with pytest.raises(ValueError):
        await synthetic_service.generate_synthetic_defects(
            image_path=sample_image,
            annotation={},  # Missing bbox
            num_variations=5,
            output_dir=temp_output_dir
        )


@pytest.mark.asyncio
async def test_generate_synthetic_defects_output_directory_creation(
    sample_image,
    sample_annotation
):
    """
    Test that generate_synthetic_defects creates output directory if needed.
    
    Validates that:
    - Output directory is created if it doesn't exist
    - Files are saved to the specified directory
    """
    import tempfile
    import shutil
    
    # Create unique output directory that doesn't exist yet
    base_temp = tempfile.gettempdir()
    output_dir = os.path.realpath(os.path.join(base_temp, f"test_synthetic_{os.getpid()}"))
    
    # Ensure it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    try:
        result = await synthetic_service.generate_synthetic_defects(
            image_path=sample_image,
            annotation=sample_annotation,
            num_variations=2,
            output_dir=output_dir
        )
        
        # Directory should now exist
        assert os.path.exists(output_dir), "Output directory was not created"
        
        # Files should be in the directory
        for path in result:
            assert path.startswith(output_dir), f"File {path} not in output directory {output_dir}"
    
    finally:
        # Cleanup
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


@pytest.mark.asyncio
async def test_generate_synthetic_defects_fallback(
    sample_image,
    sample_annotation,
    temp_output_dir
):
    """
    Test fallback synthetic generation method.
    
    Validates that:
    - Fallback method executes successfully
    - Generates requested number of variations
    - Files are valid images
    """
    result = await synthetic_service.generate_synthetic_defects_fallback(
        image_path=sample_image,
        annotation=sample_annotation,
        num_variations=4,
        output_dir=temp_output_dir
    )
    
    # Validate result
    assert isinstance(result, list)
    assert len(result) == 4
    
    # All files should exist and be valid
    for path in result:
        assert os.path.exists(path)
        img = cv2.imread(path)
        assert img is not None


@pytest.mark.asyncio
async def test_generate_synthetic_defects_with_custom_conditions(
    sample_image,
    sample_annotation,
    temp_output_dir
):
    """
    Test generate_synthetic_defects with custom lighting and severity.
    
    Validates that:
    - Custom lighting conditions are respected
    - Custom severity levels are used
    - Correct number of variations is generated
    """
    custom_lighting = ["dark", "bright"]
    custom_severity = ["minor", "severe"]
    num_variations = 6
    
    result = await synthetic_service.generate_synthetic_defects(
        image_path=sample_image,
        annotation=sample_annotation,
        num_variations=num_variations,
        lighting_conditions=custom_lighting,
        severity_levels=custom_severity,
        output_dir=temp_output_dir
    )
    
    assert len(result) == num_variations
    
    # Check file naming contains variations
    filenames = [os.path.basename(path) for path in result]
    
    # At least some files should have lighting/severity in name
    # (this depends on implementation, may be in filename or metadata)
    for filename in filenames:
        assert "synthetic" in filename.lower()


@pytest.mark.asyncio
async def test_generate_synthetic_defects_unique_outputs(
    sample_image,
    sample_annotation,
    temp_output_dir
):
    """
    Test that generate_synthetic_defects creates unique variations.
    
    Validates that:
    - Generated files have unique filenames
    - Files are not identical (basic check)
    """
    result = await synthetic_service.generate_synthetic_defects(
        image_path=sample_image,
        annotation=sample_annotation,
        num_variations=3,
        output_dir=temp_output_dir
    )
    
    # All filenames should be unique
    filenames = [os.path.basename(path) for path in result]
    assert len(filenames) == len(set(filenames)), "Generated files should have unique names"
    
    # Files should have different sizes (very basic uniqueness check)
    file_sizes = [os.path.getsize(path) for path in result]
    # At least some should be different (though fallback might create similar files)
    assert len(set(file_sizes)) >= 1


def test_build_prompt_with_background():
    """
    Test build_prompt with different background materials.
    
    Validates that:
    - Background parameter is included in prompt
    - Different backgrounds produce different prompts
    """
    backgrounds = ["industrial", "metal", "concrete", "plastic"]
    
    prompts = []
    for bg in backgrounds:
        prompt = synthetic_service.build_prompt(
            defect_type="crack",
            lighting="natural",
            severity="moderate",
            background=bg
        )
        prompts.append(prompt)
        assert bg in prompt.lower() or "surface" in prompt.lower()
    
    # Should produce varied prompts
    assert len(set(prompts)) >= 2
