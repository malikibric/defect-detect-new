"""
Quality Assurance router for AI-driven annotation validation.

This module exposes REST API endpoints for validating human annotations
using YOLO26 defect detection and providing QA feedback.
"""

import time
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from models.schemas import QACheckRequest, QACheckResponse, Annotation

from services import qa_service
from core.deps import get_current_active_user
from core.asset_resolver import resolve_image_reference
from models.db_models import User
from models.database import get_db
from core.path_security import safe_display_path
from core.serialization import annotations_from_dicts, annotations_to_dicts

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/check",
    response_model=QACheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Run QA check on annotations",
    description="""
    Validate human annotations using AI-driven quality assurance with YOLO26.
    
    This endpoint runs independent defect detection using YOLO26 and compares
    the results with human annotations to identify potential issues.
    
    **How it works:**
    1. Runs YOLO26 inference on the image independently
    2. Compares YOLO predictions with human annotations using IoU
    3. Identifies three categories of feedback:
       - **Missed Defects**: YOLO found defects that humans didn't annotate
       - **Size Warnings**: Human annotations with bounding box size deviating >40% from median
       - **Confirmed**: Annotations that match YOLO predictions (high confidence)
    
    **Use cases:**
    - Quality control for manual annotation workflows
    - Training data validation before model training
    - Identifying annotation inconsistencies
    - Catching overlooked defects
    
    **Parameters:**
    - `image_path`: Path to the image file on the server
    - `human_annotations`: List of human-provided annotations to validate
    - `iou_threshold`: Minimum IoU (0.0-1.0) to consider a match (default: 0.5)
    
    **Returns:**
    - `missed_defects`: Defects detected by AI but not by humans
    - `size_warnings`: Annotations with unusual bounding box sizes
    - `confirmed`: Annotations validated by AI
    - `total_human_annotations`: Count of human annotations
    - `total_ai_detections`: Count of AI detections
    - `processing_time_seconds`: Processing time
    
    **Interpreting Results:**
    - High number of missed_defects → Annotator may need retraining
    - Many size_warnings → Annotation guidelines may be unclear
    - High confirmed rate → Good annotation quality
    """
)
async def check_annotations(
    request: QACheckRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> QACheckResponse:
    """
    Run AI-driven QA check on human annotations.
    
    Args:
        request: QACheckRequest with image path and human annotations
        
    Returns:
        QACheckResponse with QA validation results
        
    Raises:
        HTTPException 404: If image file is not found
        HTTPException 500: If QA check fails
    """
    start_time = time.time()
    
    try:
        resolved_image_path = await resolve_image_reference(
            db=db,
            owner_id=current_user.id,
            image_path=request.image_path,
            image_asset_id=request.image_asset_id,
            source_uri=request.source_uri,
        )

        logger.info(
            "User %s running QA check for %s",
            current_user.id,
            safe_display_path(resolved_image_path),
        )
        logger.info(f"Human annotations: {len(request.human_annotations)}, IoU threshold: {request.iou_threshold}")
        
        # Call service
        qa_result = await qa_service.run_qa_check(
            image_path=resolved_image_path,
            human_annotations=annotations_to_dicts(request.human_annotations),
            iou_threshold=request.iou_threshold
        )
        
        missed_defects = annotations_from_dicts(qa_result["missed_defects"])
        confirmed = annotations_from_dicts(qa_result["confirmed"])
        
        processing_time = time.time() - start_time
        
        logger.info(f"QA check successful in {processing_time:.2f}s")
        logger.info(f"  Confirmed: {len(confirmed)}, Missed: {len(missed_defects)}, Warnings: {len(qa_result['size_warnings'])}")
        
        return QACheckResponse(
            missed_defects=missed_defects,
            size_warnings=qa_result["size_warnings"],
            confirmed=confirmed,
            total_human_annotations=qa_result["total_human_annotations"],
            total_ai_detections=qa_result["total_ai_detections"],
            processing_time_seconds=processing_time
        )
        
    except FileNotFoundError as e:
        logger.error(f"Image file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found: {str(e)}"
        )
    
    except RuntimeError as e:
        logger.error(f"QA check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QA check failed: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in QA check: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QA check failed: {str(e)}"
        )
