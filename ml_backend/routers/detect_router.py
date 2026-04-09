# FIXED: New convenience router for frontend upload-and-detect workflow
"""
Convenience detection endpoints for direct file upload with ML inference.

These endpoints accept image uploads and run ML models in a single request,
without requiring authentication. Designed for the frontend's direct
upload-and-detect workflow.
"""

import json
import logging
import os
import tempfile
import time
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from services import qa_service, sam_service, patch_service

logger = logging.getLogger(__name__)

router = APIRouter()


async def _save_temp_file(upload: UploadFile) -> str:
    """Save an uploaded file to a temporary path and return it."""
    suffix = os.path.splitext(upload.filename or "upload.jpg")[1] or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        contents = await upload.read()
        os.write(fd, contents)
    finally:
        os.close(fd)
    return path


def _cleanup(path: str | None) -> None:
    """Remove a temporary file, ignoring errors."""
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


@router.post("/upload-image")
async def upload_and_detect(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload an image and run YOLO defect detection in one step.

    Returns detections in the format expected by the frontend:
    { detections: [...], total_detections, processing_time_seconds }
    """
    start_time = time.time()
    temp_path = None

    try:
        temp_path = await _save_temp_file(file)

        from services.storage_service import load_cv2_image

        image, _ = await load_cv2_image(temp_path)

        model = qa_service.load_yolo_model()
        results = model(image, verbose=False)
        raw_detections = qa_service.extract_yolo26_detections(
            results=results, names=getattr(model, "names", None)
        )

        detections = [
            {
                "id": f"{det['class_name']}-{i}",
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "source": "AI Detected",
                "status": "pending",
            }
            for i, det in enumerate(raw_detections)
        ]

        return {
            "detections": detections,
            "total_detections": len(detections),
            "processing_time_seconds": time.time() - start_time,
        }

    except Exception as exc:
        logger.error("Detection failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {exc}",
        )
    finally:
        _cleanup(temp_path)


@router.post("/detect/propagate")
async def propagate_simple(
    file: UploadFile = File(...),
    seed_annotations: str = Form(...),
    similarity_threshold: float = Form(default=0.75),
) -> Dict[str, Any]:
    """SAM label propagation from uploaded image and seed annotations."""
    start_time = time.time()
    temp_path = None

    try:
        temp_path = await _save_temp_file(file)
        seeds = json.loads(seed_annotations)

        proposed = await sam_service.propagate_annotations(
            image_path=temp_path,
            seed_annotations=seeds,
            similarity_threshold=similarity_threshold,
        )

        return {
            "proposed_annotations": proposed,
            "total_proposed": len(proposed),
            "processing_time_seconds": time.time() - start_time,
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in seed_annotations",
        )
    except Exception as exc:
        logger.error("Propagation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Propagation failed: {exc}",
        )
    finally:
        _cleanup(temp_path)


@router.post("/detect/qa-check")
async def qa_check_simple(
    file: UploadFile = File(...),
    annotations: str = Form(...),
    iou_threshold: float = Form(default=0.5),
) -> Dict[str, Any]:
    """QA check from uploaded image and current annotations."""
    temp_path = None

    try:
        temp_path = await _save_temp_file(file)
        human_annotations = json.loads(annotations)

        result = await qa_service.run_qa_check(
            image_path=temp_path,
            human_annotations=human_annotations,
            iou_threshold=iou_threshold,
        )
        return result

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in annotations",
        )
    except Exception as exc:
        logger.error("QA check failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QA check failed: {exc}",
        )
    finally:
        _cleanup(temp_path)


@router.post("/detect/extract-patches")
async def extract_patches_simple(
    file: UploadFile = File(...),
    annotations: str = Form(...),
    patch_size: str = Form(default=""),
    padding_factor: float = Form(default=1.5),
) -> Dict[str, Any]:
    """Patch extraction from uploaded image and annotations."""
    temp_path = None

    try:
        temp_path = await _save_temp_file(file)
        annotation_list = json.loads(annotations)

        effective_patch_size = None
        if patch_size and patch_size.strip():
            try:
                ps = int(patch_size)
                if ps > 0:
                    effective_patch_size = ps
            except ValueError:
                pass

        result = await patch_service.extract_patches(
            image_path=temp_path,
            annotations=annotation_list,
            patch_size=effective_patch_size,
            padding_factor=padding_factor,
        )

        return {
            "patches": result["patches"],
            "optimal_patch_size": result["optimal_patch_size"],
            "total_patches": result["total_patches"],
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in annotations",
        )
    except Exception as exc:
        logger.error("Patch extraction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Patch extraction failed: {exc}",
        )
    finally:
        _cleanup(temp_path)
