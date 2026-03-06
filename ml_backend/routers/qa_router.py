"""
Quality Assurance router for AI-driven annotation validation.

This module exposes REST API endpoints for validating human annotations
using YOLO26 defect detection and providing QA feedback.
"""

import time
import logging
from fastapi import APIRouter, HTTPException, status
from models.schemas import QACheckRequest, QACheckResponse, Annotation

from services import qa_service

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
async def check_annotations(request: QACheckRequest) -> QACheckResponse:
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
        logger.info(f"Running QA check for {request.image_path}")
        logger.info(f"Human annotations: {len(request.human_annotations)}, IoU threshold: {request.iou_threshold}")
        
        # Convert Pydantic models to dicts
        human_annotations_dict = [
            {
                "bbox": ann.bbox,
                "class_name": ann.class_name,
                "confidence": ann.confidence,
                "annotation_id": ann.annotation_id
            }
            for ann in request.human_annotations
        ]
        
        # Call service
        qa_result = await qa_service.run_qa_check(
            image_path=request.image_path,
            human_annotations=human_annotations_dict,
            iou_threshold=request.iou_threshold
        )
        
        # Convert results to Pydantic models
        missed_defects = [
            Annotation(
                bbox=det["bbox"],
                class_name=det["class_name"],
                confidence=det.get("confidence"),
                annotation_id=det.get("annotation_id")
            )
            for det in qa_result["missed_defects"]
        ]
        
        confirmed = [
            Annotation(
                bbox=ann["bbox"],
                class_name=ann["class_name"],
                confidence=ann.get("confidence"),
                annotation_id=ann.get("annotation_id")
            )
            for ann in qa_result["confirmed"]
        ]
        
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
