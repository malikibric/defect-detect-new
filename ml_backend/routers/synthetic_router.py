"""
Synthetic data generation router for creating defect variations.

This module exposes REST API endpoints for generating synthetic defect
variations using Stable Diffusion inpainting.
"""

import time
import logging
from fastapi import APIRouter, HTTPException, status
from models.schemas import SyntheticGenerateRequest, SyntheticGenerateResponse

from services import synthetic_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/generate",
    response_model=SyntheticGenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate synthetic defect variations",
    description="""
    Generate realistic synthetic variations of defects using Stable Diffusion inpainting.
    
    This endpoint creates diverse, high-quality defect variations by manipulating
    lighting conditions, defect severity, and background characteristics while
    maintaining photorealistic quality.
    
    **How it works:**
    
    1. **Image Preparation**:
       - Loads the source image
       - Extracts the defect region from the annotation bounding box
       - Creates an inpainting mask with slight expansion for natural blending
    
    2. **Prompt Engineering**:
       - Builds detailed prompts for each variation
       - Combines: defect_type + lighting + severity + background
       - Examples:
         - "A severe crack defect on concrete, bright lighting, photorealistic"
         - "A minor corrosion defect on metal, dim lighting, shadows"
    
    3. **Stable Diffusion Inpainting**:
       - Runs the diffusion model with the constructed prompt
       - Inpaints only the masked region (preserves surrounding context)
       - Uses different random seeds for variation diversity
       - Applies guidance scale 7.5 for prompt adherence
    
    4. **Variation Strategy**:
       - Cycles through lighting conditions (dark, bright, side-lit)
       - Cycles through severity levels (minor, moderate, severe)
       - Generates unique combinations for maximum diversity
    
    **Use cases:**
    - **Data Augmentation**: Expand training datasets with rare defect types
    - **Class Balancing**: Generate more examples of underrepresented defects
    - **Scenario Testing**: Create defects under various lighting conditions
    - **Model Robustness**: Train models on diverse defect appearances
    
    **Best Practices:**
    - Start with 10-15 variations per defect
    - Use diverse lighting conditions for robust models
    - Review generated images before adding to training set
    - Combine with real data (don't train only on synthetic)
    
    **Parameters:**
    - `image_path`: Path to the source image file on the server
    - `annotation`: Single defect annotation to generate variations from
    - `num_variations`: Number of variations to generate (1-50, default: 10)
    - `lighting_conditions`: List of lighting types (default: ["dark", "bright", "side-lit"])
    - `severity_levels`: List of severity levels (default: ["minor", "moderate", "severe"])
    
    **Returns:**
    - `generated_images`: List of file paths to generated variations
    - `total_generated`: Count of successfully generated images
    - `output_directory`: Directory containing all generated images
    - `processing_time_seconds`: Total processing time
    
    **Note:** Generation can take 5-10 seconds per image on GPU, longer on CPU.
    Consider using background tasks for large batch requests.
    """
)
async def generate_synthetic_variations(
    request: SyntheticGenerateRequest
) -> SyntheticGenerateResponse:
    """
    Generate synthetic defect variations using Stable Diffusion.
    
    Args:
        request: SyntheticGenerateRequest with source image and annotation
        
    Returns:
        SyntheticGenerateResponse with paths to generated images
        
    Raises:
        HTTPException 404: If source image is not found
        HTTPException 422: If annotation is invalid
        HTTPException 500: If generation fails
    """
    start_time = time.time()
    
    try:
        logger.info(f"Generating synthetic variations for {request.image_path}")
        logger.info(f"Requested variations: {request.num_variations}")
        logger.info(f"Lighting conditions: {request.lighting_conditions}")
        logger.info(f"Severity levels: {request.severity_levels}")
        
        # Convert Pydantic model to dict
        annotation_dict = {
            "bbox": request.annotation.bbox,
            "class_name": request.annotation.class_name,
            "confidence": request.annotation.confidence,
            "annotation_id": request.annotation.annotation_id
        }
        
        # Call service
        generated_paths = await synthetic_service.generate_synthetic_defects(
            image_path=request.image_path,
            annotation=annotation_dict,
            num_variations=request.num_variations,
            lighting_conditions=request.lighting_conditions,
            severity_levels=request.severity_levels
        )
        
        processing_time = time.time() - start_time
        
        # Determine output directory from first generated path
        output_dir = "output/synthetic"
        if generated_paths:
            from pathlib import Path
            output_dir = str(Path(generated_paths[0]).parent)
        
        logger.info(f"Generation successful: {len(generated_paths)} images in {processing_time:.2f}s")
        logger.info(f"Average time per image: {processing_time/max(len(generated_paths), 1):.2f}s")
        
        return SyntheticGenerateResponse(
            generated_images=generated_paths,
            total_generated=len(generated_paths),
            output_directory=output_dir,
            processing_time_seconds=processing_time
        )
        
    except FileNotFoundError as e:
        logger.error(f"Source image not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source image not found: {str(e)}"
        )
    
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input: {str(e)}"
        )
    
    except RuntimeError as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthetic generation failed: {str(e)}"
        )
