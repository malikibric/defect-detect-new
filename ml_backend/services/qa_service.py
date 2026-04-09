"""
Quality Assurance service using YOLO26 for defect detection validation.

This module provides AI-driven QA capabilities to validate human annotations
by comparing them with YOLO26 predictions and identifying potential issues.
"""

import os
import time
import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Global model registry (shared application state)
from core.state import model_registry
from core.geometry import calculate_iou as _calculate_iou
from services.storage_service import load_cv2_image


def load_yolo_model() -> Any:
    """
    Load YOLO26 model for defect detection.
    
    This function lazy-loads the YOLO model on first use. It attempts to load
    a configured YOLO26 checkpoint, falling back to the pretrained YOLO26 Nano
    model (``yolo26n.pt``) for cloud memory efficiency.

    YOLO26 uses an NMS-free, end-to-end detection architecture and STAL
    (Small-Target-Aware Label Assignment), which improves minor-defect recall
    in dense or fine-grained industrial surfaces.
    
    Returns:
        Loaded YOLO model instance
        
    Raises:
        RuntimeError: If model loading fails
    """
    if model_registry.get("yolo_model") is None:
        logger.info("Loading YOLO26 model...")
        
        try:
            from ultralytics import YOLO
            
            # Prefer configured checkpoint, defaulting to YOLO26 Nano.
            configured_model_path = os.getenv("YOLO_MODEL", "models/yolo26n.pt")
            fallback_model_name = "yolo26n.pt"
            
            if os.path.exists(configured_model_path):
                logger.info("Loading configured YOLO26 model from %s", configured_model_path)
                model = YOLO(configured_model_path)
            else:
                # Fall back to pretrained YOLO26 Nano (small footprint for cloud workers).
                logger.info(
                    "Configured model not found. Loading pretrained YOLO26 Nano model: %s",
                    fallback_model_name,
                )
                model = YOLO(fallback_model_name)
            
            # Move to appropriate device
            device = model_registry.get("device", "cpu")
            model.to(device)
            
            model_registry["yolo_model"] = model
            logger.info("YOLO26 model loaded successfully on %s", device)
            
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            raise RuntimeError(f"YOLO model loading failed: {e}")
    
    return model_registry["yolo_model"]


def _to_float(value: Any) -> float:
    """Convert a scalar/tensor-like value to a Python float safely."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        raise ValueError(f"Expected scalar tensor/array, got shape {value.shape}")
    return float(value)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensor-like value to a numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_yolo26_detections(results: Any, names: Dict[int, str] | List[str] | None) -> List[Dict[str, Any]]:
    """
    Extract normalized detections from YOLO26 inference output.

    Supports both structured outputs (``boxes.xyxy``, ``boxes.conf``, ``boxes.cls``)
    and row-based outputs (``boxes.data`` with columns ``x1,y1,x2,y2,conf,cls``).
    This keeps parsing robust across Ultralytics output variants for NMS-free heads.

    Args:
        results: Iterable of Ultralytics result objects.
        names: Class id to class-name mapping from the model.

    Returns:
        List of detections in unified schema.
    """
    detections: List[Dict[str, Any]] = []

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        # Preferred path: explicit xyxy/conf/cls fields.
        if hasattr(boxes, "xyxy") and hasattr(boxes, "conf") and hasattr(boxes, "cls"):
            xyxy_array = _to_numpy(boxes.xyxy)
            conf_array = _to_numpy(boxes.conf).reshape(-1)
            cls_array = _to_numpy(boxes.cls).reshape(-1)

            for index in range(len(xyxy_array)):
                x1, y1, x2, y2 = [float(coord) for coord in xyxy_array[index][:4]]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                confidence = _to_float(conf_array[index])
                class_id = int(_to_float(cls_array[index]))

                if isinstance(names, dict):
                    class_name = str(names.get(class_id, class_id))
                elif isinstance(names, list) and 0 <= class_id < len(names):
                    class_name = str(names[class_id])
                else:
                    class_name = str(class_id)

                detections.append(
                    {
                        "bbox": [x1, y1, width, height],
                        "class_name": class_name,
                        "confidence": confidence,
                        "class_id": class_id,
                    }
                )
            continue

        # Fallback path: generic Nx6+ ``boxes.data`` format.
        if hasattr(boxes, "data"):
            data_array = _to_numpy(boxes.data)
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)

            for row in data_array:
                if row.shape[0] < 6:
                    continue
                x1, y1, x2, y2 = [float(coord) for coord in row[:4]]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                confidence = float(row[4])
                class_id = int(row[5])

                if isinstance(names, dict):
                    class_name = str(names.get(class_id, class_id))
                elif isinstance(names, list) and 0 <= class_id < len(names):
                    class_name = str(names[class_id])
                else:
                    class_name = str(class_id)

                detections.append(
                    {
                        "bbox": [x1, y1, width, height],
                        "class_name": class_name,
                        "confidence": confidence,
                        "class_id": class_id,
                    }
                )

    return detections


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    IoU is a metric for measuring overlap between two bounding boxes.
    It is calculated as the area of intersection divided by the area of union.
    
    Formula:
        IoU = Area(box1 ∩ box2) / Area(box1 ∪ box2)
    
    Args:
        box1: First bounding box in format [x, y, w, h]
        box2: Second bounding box in format [x, y, w, h]
        
    Returns:
        IoU score between 0.0 (no overlap) and 1.0 (perfect overlap)
        
    Example:
        >>> box1 = [10, 10, 50, 50]  # x=10, y=10, width=50, height=50
        >>> box2 = [30, 30, 50, 50]  # Overlapping box
        >>> iou = calculate_iou(box1, box2)
        >>> print(f"IoU: {iou:.2f}")
        IoU: 0.14
    """
    return _calculate_iou(box1, box2)


def calculate_box_area(bbox: List[float]) -> float:
    """
    Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box in format [x, y, w, h]
        
    Returns:
        Area of the bounding box
    """
    return bbox[2] * bbox[3]


async def run_qa_check(
    image_path: str,
    human_annotations: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Run AI-driven quality assurance check on human annotations.
    
    This function validates human annotations by:
    1. Running independent YOLO26 inference on the image
    2. Comparing YOLO26 predictions with human annotations using IoU
    3. Identifying missed defects (YOLO26 found, human didn't)
    4. Flagging size warnings (human annotation significantly differs from YOLO26)
    5. Confirming correct annotations (high IoU with YOLO26 predictions)
    
    **QA Report Components:**
    
    - **missed_defects**: Defects detected by YOLO26 but not annotated by humans.
      These represent potential annotation errors or overlooked defects.
      
        - **size_warnings**: Human annotations where the bounding box size deviates
            more than 40% from the median YOLO26 box size for that class. This indicates
      potential sizing errors in annotation.
      
        - **confirmed**: Annotations that match YOLO26 predictions with IoU > threshold.
            These are validated as correct by the AI model.

        YOLO26 uses STAL for stronger small-target supervision, so this function
        keeps bounding boxes in floating-point precision to preserve tiny defect
        geometry during IoU and size-deviation checks.
    
    Args:
        image_path: Path to the image file to analyze
        human_annotations: List of human-provided annotations with bbox and class_name
        iou_threshold: Minimum IoU score to consider a match (default: 0.5)
        
    Returns:
        Dictionary containing:
            - missed_defects: List of YOLO26 detections not covered by humans
            - size_warnings: List of annotations with size deviation warnings
            - confirmed: List of annotations confirmed by YOLO26
            - total_human_annotations: Count of human annotations
            - total_ai_detections: Count of YOLO26 detections
            - processing_time_seconds: Time taken for QA check
            
    Raises:
        FileNotFoundError: If image file does not exist
        RuntimeError: If YOLO26 inference fails
    """
    start_time = time.time()
    
    # Validate inputs
    image, display_image_path = await load_cv2_image(image_path)
    
    logger.info(
        "Running QA check on %s (human_annotations=%s, iou_threshold=%.2f)",
        display_image_path,
        len(human_annotations),
        iou_threshold,
    )
    
    # Load YOLO26 model
    model = load_yolo_model()
    
    # Run YOLO26 inference. Output is parsed without NMS assumptions.
    logger.info("Running YOLO26 inference...")
    try:
        results = model(image, verbose=False)
        yolo_detections = extract_yolo26_detections(results=results, names=getattr(model, "names", None))

        logger.info("YOLO26 found %d detections", len(yolo_detections))
        
    except Exception as e:
        logger.error("YOLO26 inference failed: %s", e)
        raise RuntimeError(f"YOLO26 inference failed: {e}")
    
    # Initialize QA report components
    missed_defects = []
    size_warnings = []
    confirmed = []
    
    # Track which YOLO26 detections have been matched
    matched_yolo_indices = set()
    matched_human_indices = set()
    confirmed_matches: Dict[int, tuple[int, float]] = {}
    
    # Step 1: Build globally sorted IoU matches so one human annotation cannot
    # steal the best detection from another annotation.
    candidate_matches: List[tuple[float, int, int]] = []
    for human_idx, human_ann in enumerate(human_annotations):
        human_bbox = human_ann["bbox"]
        for yolo_idx, yolo_det in enumerate(yolo_detections):
            iou = calculate_iou(human_bbox, yolo_det["bbox"])
            if iou >= iou_threshold:
                candidate_matches.append((iou, human_idx, yolo_idx))

    candidate_matches.sort(key=lambda match: match[0], reverse=True)

    for iou, human_idx, yolo_idx in candidate_matches:
        if human_idx in matched_human_indices or yolo_idx in matched_yolo_indices:
            continue
        matched_human_indices.add(human_idx)
        matched_yolo_indices.add(yolo_idx)
        confirmed_matches[human_idx] = (yolo_idx, iou)

    # Step 2: Materialize confirmed annotations from the chosen one-to-one matches.
    for human_idx, human_ann in enumerate(human_annotations):
        match = confirmed_matches.get(human_idx)
        if match is None:
            continue

        best_yolo_idx, best_iou = match
        human_class = human_ann.get("class_name", "defect")

        confirmed.append({
            "bbox": human_ann["bbox"],
            "class_name": human_class,
            "confidence": yolo_detections[best_yolo_idx]["confidence"],
            "iou_with_yolo": best_iou,
            "annotation_id": human_ann.get("annotation_id", f"human_{human_idx}")
        })
    
    # Step 3: Find missed defects (YOLO26 detected, human didn't)
    for yolo_idx, yolo_det in enumerate(yolo_detections):
        if yolo_idx not in matched_yolo_indices:
            # This YOLO26 detection was not matched with any human annotation
            missed_defects.append({
                "bbox": yolo_det["bbox"],
                "class_name": yolo_det["class_name"],
                "confidence": yolo_det["confidence"],
                "annotation_id": f"yolo_missed_{yolo_idx}"
            })
    
    # Step 4: Check for size warnings
    # Calculate median YOLO26 box size per class
    class_sizes: Dict[str, List[float]] = {}
    for yolo_det in yolo_detections:
        class_name = yolo_det["class_name"]
        area = calculate_box_area(yolo_det["bbox"])
        class_sizes.setdefault(class_name, []).append(area)
    
    # Calculate median for each class
    class_median_sizes = {
        cls: np.median(sizes) for cls, sizes in class_sizes.items()
    }
    
    # Check human annotations for size deviations
    for human_idx, human_ann in enumerate(human_annotations):
        human_bbox = human_ann["bbox"]
        human_class = human_ann.get("class_name", "defect")
        human_area = calculate_box_area(human_bbox)
        
        # Get median size for this class
        median_size = class_median_sizes.get(human_class)
        
        if median_size is not None and median_size > 0:
            # Calculate deviation percentage
            deviation = abs(human_area - median_size) / median_size
            
            # Flag if deviation > 40%
            if deviation > 0.4:
                size_warnings.append({
                    "bbox": human_bbox,
                    "class_name": human_class,
                    "annotation_id": human_ann.get("annotation_id", f"human_{human_idx}"),
                    "human_area": human_area,
                    "median_area": median_size,
                    "deviation_percentage": deviation * 100,
                    "warning": f"Size deviates {deviation*100:.1f}% from median"
                })
    
    processing_time = time.time() - start_time
    
    logger.info(
        "QA check complete in %.2fs (confirmed=%s, missed=%s, size_warnings=%s)",
        processing_time,
        len(confirmed),
        len(missed_defects),
        len(size_warnings),
    )
    
    return {
        "missed_defects": missed_defects,
        "size_warnings": size_warnings,
        "confirmed": confirmed,
        "total_human_annotations": len(human_annotations),
        "total_ai_detections": len(yolo_detections),
        "processing_time_seconds": processing_time
    }
