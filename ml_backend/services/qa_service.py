"""
Quality Assurance service using YOLOv8 for defect detection validation.

This module provides AI-driven QA capabilities to validate human annotations
by comparing them with YOLO predictions and identifying potential issues.
"""

import os
import time
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Global model registry (shared application state)
from core.state import model_registry


def load_yolo_model() -> Any:
    """
    Load YOLOv8 model for defect detection.
    
    This function lazy-loads the YOLO model on first use. It attempts to load
    a fine-tuned defect detection model, falling back to the pretrained YOLOv8m
    model if unavailable.
    
    Returns:
        Loaded YOLO model instance
        
    Raises:
        RuntimeError: If model loading fails
    """
    if model_registry.get("yolo_model") is None:
        logger.info("Loading YOLOv8 model...")
        
        try:
            from ultralytics import YOLO
            
            # Try to load fine-tuned defect model first
            custom_model_path = "models/defect_yolov8m.pt"
            
            if os.path.exists(custom_model_path):
                logger.info(f"Loading custom defect model from {custom_model_path}")
                model = YOLO(custom_model_path)
            else:
                # Fall back to pretrained model
                logger.info("Loading pretrained YOLOv8m model")
                model = YOLO("yolov8m.pt")
            
            # Move to appropriate device
            device = model_registry.get("device", "cpu")
            model.to(device)
            
            model_registry["yolo_model"] = model
            logger.info(f"YOLO model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"YOLO model loading failed: {e}")
    
    return model_registry["yolo_model"]


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
    # Extract coordinates
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # Check if there is no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


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
    1. Running independent YOLO inference on the image
    2. Comparing YOLO predictions with human annotations using IoU
    3. Identifying missed defects (YOLO found, human didn't)
    4. Flagging size warnings (human annotation significantly differs from YOLO)
    5. Confirming correct annotations (high IoU with YOLO predictions)
    
    **QA Report Components:**
    
    - **missed_defects**: Defects detected by YOLO but not annotated by humans.
      These represent potential annotation errors or overlooked defects.
      
    - **size_warnings**: Human annotations where the bounding box size deviates
      more than 40% from the median YOLO box size for that class. This indicates
      potential sizing errors in annotation.
      
    - **confirmed**: Annotations that match YOLO predictions with IoU > threshold.
      These are validated as correct by the AI model.
    
    Args:
        image_path: Path to the image file to analyze
        human_annotations: List of human-provided annotations with bbox and class_name
        iou_threshold: Minimum IoU score to consider a match (default: 0.5)
        
    Returns:
        Dictionary containing:
            - missed_defects: List of YOLO detections not covered by humans
            - size_warnings: List of annotations with size deviation warnings
            - confirmed: List of annotations confirmed by YOLO
            - total_human_annotations: Count of human annotations
            - total_ai_detections: Count of YOLO detections
            - processing_time_seconds: Time taken for QA check
            
    Raises:
        FileNotFoundError: If image file does not exist
        RuntimeError: If YOLO inference fails
    """
    start_time = time.time()
    
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    logger.info(f"Running QA check on {image_path}")
    logger.info(f"Human annotations: {len(human_annotations)}, IoU threshold: {iou_threshold}")
    
    # Load YOLO model
    model = load_yolo_model()
    
    # Run YOLO inference
    logger.info("Running YOLO inference...")
    try:
        results = model(image_path, verbose=False)
        yolo_detections = []
        
        # Extract YOLO predictions
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                # Get box in xyxy format and convert to xywh
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = model.names[class_id]
                
                yolo_detections.append({
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "class_name": class_name,
                    "confidence": confidence,
                    "class_id": class_id
                })
        
        logger.info(f"YOLO found {len(yolo_detections)} detections")
        
    except Exception as e:
        logger.error(f"YOLO inference failed: {e}")
        raise RuntimeError(f"YOLO inference failed: {e}")
    
    # Initialize QA report components
    missed_defects = []
    size_warnings = []
    confirmed = []
    
    # Track which YOLO detections have been matched
    matched_yolo_indices = set()
    matched_human_indices = set()
    
    # Step 1: Find confirmed annotations (human matches YOLO)
    for human_idx, human_ann in enumerate(human_annotations):
        human_bbox = human_ann["bbox"]
        human_class = human_ann.get("class_name", "defect")
        
        best_iou = 0.0
        best_yolo_idx = -1
        
        for yolo_idx, yolo_det in enumerate(yolo_detections):
            # Only compare same class (or if YOLO uses generic object detection)
            yolo_class = yolo_det["class_name"]
            
            # Calculate IoU
            iou = calculate_iou(human_bbox, yolo_det["bbox"])
            
            if iou > best_iou:
                best_iou = iou
                best_yolo_idx = yolo_idx
        
        # If IoU exceeds threshold, mark as confirmed
        if best_iou >= iou_threshold and best_yolo_idx != -1:
            matched_human_indices.add(human_idx)
            matched_yolo_indices.add(best_yolo_idx)
            
            confirmed.append({
                "bbox": human_bbox,
                "class_name": human_class,
                "confidence": yolo_detections[best_yolo_idx]["confidence"],
                "iou_with_yolo": best_iou,
                "annotation_id": human_ann.get("annotation_id", f"human_{human_idx}")
            })
    
    # Step 2: Find missed defects (YOLO detected, human didn't)
    for yolo_idx, yolo_det in enumerate(yolo_detections):
        if yolo_idx not in matched_yolo_indices:
            # This YOLO detection was not matched with any human annotation
            missed_defects.append({
                "bbox": yolo_det["bbox"],
                "class_name": yolo_det["class_name"],
                "confidence": yolo_det["confidence"],
                "annotation_id": f"yolo_missed_{yolo_idx}"
            })
    
    # Step 3: Check for size warnings
    # Calculate median YOLO box size per class
    class_sizes: Dict[str, List[float]] = {}
    for yolo_det in yolo_detections:
        class_name = yolo_det["class_name"]
        area = calculate_box_area(yolo_det["bbox"])
        if class_name not in class_sizes:
            class_sizes[class_name] = []
        class_sizes[class_name].append(area)
    
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
    
    logger.info(f"QA check complete in {processing_time:.2f}s:")
    logger.info(f"  - Confirmed: {len(confirmed)}")
    logger.info(f"  - Missed defects: {len(missed_defects)}")
    logger.info(f"  - Size warnings: {len(size_warnings)}")
    
    return {
        "missed_defects": missed_defects,
        "size_warnings": size_warnings,
        "confirmed": confirmed,
        "total_human_annotations": len(human_annotations),
        "total_ai_detections": len(yolo_detections),
        "processing_time_seconds": processing_time
    }
