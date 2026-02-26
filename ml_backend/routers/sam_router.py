"""
SAM router for automated label propagation endpoints.

This module exposes REST API endpoints for using the Segment Anything Model
to propagate annotations across images with few-shot learning.
"""

import time
import logging
from fastapi import APIRouter, HTTPException, status
from models.schemas import SAMPropagateRequest, SAMPropagateResponse, Annotation

from services import sam_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/propagate",
    response_model=SAMPropagateResponse,
    status_code=status.HTTP_200_OK,
    summary="Propagate annotations using SAM",
    description="""
    Automatically propagate annotations across an image using Meta's Segment Anything Model.
    
    This endpoint takes 2-10 seed annotations and uses SAM's segmentation capabilities
    combined with few-shot learning to find similar regions throughout the image.
    
    **How it works:**
    1. Accepts seed annotations (bounding boxes with class labels)
    2. Extracts visual features from seed regions
    3. Uses SAM to generate candidate segmentation masks across the image
    4. Compares each candidate with seed examples using similarity metrics
    5. Returns only proposals with similarity above the specified threshold
    
    **Use case:**
    - Annotate 2-3 examples of a defect type manually
    - Let SAM find all similar defects in the image
    - Review and accept/reject the proposed annotations
    
    **Parameters:**
    - `image_path`: Path to the image file on the server
    - `seed_annotations`: 2-10 seed annotations with bounding boxes and class labels
    - `similarity_threshold`: Minimum similarity score (0.0-1.0) for proposals (default: 0.75)
    
    **Returns:**
    - `proposed_annotations`: List of AI-generated annotation proposals
    - `total_proposed`: Count of proposed annotations
    - `processing_time_seconds`: Time taken for processing
    """
)
async def propagate_annotations(request: SAMPropagateRequest) -> SAMPropagateResponse:
    """
    Propagate annotations using SAM with few-shot learning.
    
    Args:
        request: SAMPropagateRequest with image path and seed annotations
        
    Returns:
        SAMPropagateResponse with proposed annotations
        
    Raises:
        HTTPException 404: If image file is not found
        HTTPException 422: If seed annotations are invalid
        HTTPException 500: If processing fails
    """
    start_time = time.time()
    
    try:
        logger.info(f"Propagating annotations for {request.image_path}")
        logger.info(f"Seed annotations: {len(request.seed_annotations)}, threshold: {request.similarity_threshold}")
        
        # Convert Pydantic models to dicts
        seed_annotations_dict = [
            {
                "bbox": ann.bbox,
                "class_name": ann.class_name,
                "confidence": ann.confidence,
                "annotation_id": ann.annotation_id
            }
            for ann in request.seed_annotations
        ]
        
        # Call service
        proposed = await sam_service.propagate_annotations(
            image_path=request.image_path,
            seed_annotations=seed_annotations_dict,
            similarity_threshold=request.similarity_threshold
        )
        
        # Convert to Pydantic models
        proposed_annotations = [
            Annotation(
                bbox=prop["bbox"],
                class_name=prop["class_name"],
                confidence=prop.get("confidence"),
                annotation_id=prop.get("annotation_id")
            )
            for prop in proposed
        ]
        
        processing_time = time.time() - start_time
        
        logger.info(f"Propagation successful: {len(proposed_annotations)} proposals in {processing_time:.2f}s")
        
        return SAMPropagateResponse(
            proposed_annotations=proposed_annotations,
            total_proposed=len(proposed_annotations),
            processing_time_seconds=processing_time
        )
        
    except FileNotFoundError as e:
        logger.error(f"Image file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found: {str(e)}"
        )
    
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Propagation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Propagation failed: {str(e)}"
        )
