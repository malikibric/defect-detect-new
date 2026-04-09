"""
SAM (Segment Anything Model) service for automated label propagation.

This module provides functionality to propagate annotations across images
using Meta's Segment Anything Model with few-shot learning.
"""

import os
import time
import logging
from typing import List, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Global model registry (shared application state)
from core.state import model_registry
from core.geometry import calculate_bbox_similarity as _calculate_bbox_similarity
from services.storage_service import load_cv2_image


def load_sam_model() -> Any:
    """
    Load the Segment Anything Model (SAM) with checkpoint.
    
    This function lazy-loads the SAM model on first use to optimize
    memory usage. The model is cached in the global model_registry.
    
    Returns:
        Loaded SAM model instance
        
    Raises:
        RuntimeError: If model checkpoint file is not found
    """
    if model_registry.get("sam_model") is None:
        logger.info("Loading SAM model...")
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            # Expected checkpoint path (user should download this)
            checkpoint_path = "models/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            
            if not os.path.exists(checkpoint_path):
                logger.warning(f"SAM checkpoint not found at {checkpoint_path}")
                logger.info("Creating placeholder - download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
                # For development, we'll create a mock model
                # In production, raise error here
                model_registry["sam_model"] = None
                model_registry["sam_predictor"] = None
                return None
            
            # Load SAM model
            device = model_registry.get("device", "cpu")
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            
            # Create predictor
            predictor = SamPredictor(sam)
            
            model_registry["sam_model"] = sam
            model_registry["sam_predictor"] = predictor
            
            logger.info(f"SAM model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            model_registry["sam_model"] = None
            model_registry["sam_predictor"] = None
            return None
    
    return model_registry["sam_predictor"]


def calculate_bbox_similarity(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate similarity between two bounding boxes using IoU and size ratio.
    
    This function combines Intersection over Union (IoU) with size similarity
    to determine if two bounding boxes represent similar objects.
    
    Args:
        bbox1: First bounding box [x, y, w, h]
        bbox2: Second bounding box [x, y, w, h]
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    return _calculate_bbox_similarity(bbox1, bbox2)


def extract_features_from_bbox(image: np.ndarray, bbox: List[float]) -> np.ndarray:
    """
    Extract feature representation from a bounding box region.
    
    Args:
        image: Input image as numpy array
        bbox: Bounding box [x, y, w, h]
        
    Returns:
        Feature vector representing the region
    """
    x, y, bbox_w, bbox_h = [int(v) for v in bbox]
    
    # Ensure coordinates are within image bounds
    h_img, w_img = image.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    region_w = max(0, min(bbox_w, w_img - x))
    region_h = max(0, min(bbox_h, h_img - y))
    
    # Extract region
    region = image[y:y+region_h, x:x+region_w]
    
    if region.size == 0:
        return np.zeros(128)
    
    # Simple feature extraction: color histogram and edge density
    # Convert to grayscale for edge detection
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region
    
    # Calculate edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges) / 255.0
    
    # Calculate color histogram (if color image)
    if len(region.shape) == 3:
        hist_b = cv2.calcHist([region], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([region], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([region], [2], None, [32], [0, 256])
        color_features = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        color_features = color_features / (color_features.sum() + 1e-6)
    else:
        hist = cv2.calcHist([gray], [0], None, [96], [0, 256])
        color_features = hist.flatten()
        color_features = color_features / (color_features.sum() + 1e-6)
    
    # Combine features
    features = np.concatenate(
        [[edge_density, bbox_w, bbox_h, bbox_w / bbox_h if bbox_h > 0 else 1.0], color_features]
    )
    
    # Pad or truncate to fixed size
    if len(features) < 128:
        features = np.pad(features, (0, 128 - len(features)))
    else:
        features = features[:128]
    
    return features


async def propagate_annotations(
    image_path: str,
    seed_annotations: List[Dict[str, Any]],
    similarity_threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Propagate annotations across an image using SAM and few-shot learning.
    
    This function takes 2-3 seed annotations and uses SAM to find similar
    regions throughout the image. It applies few-shot logic by comparing
    generated masks with seed examples and only proposing regions with
    high similarity scores.
    
    Pipeline:
    1. Load image and SAM model
    2. Extract features from seed annotations
    3. Generate SAM masks using automatic mask generation
    4. Compare each mask with seed examples using similarity metrics
    5. Filter proposals by similarity threshold
    6. Return proposed annotations in COCO format
    
    Args:
        image_path: Path to the input image file
        seed_annotations: List of 2-3 seed annotations with bbox and class_name
        similarity_threshold: Minimum similarity score (0.0-1.0) for proposals
        
    Returns:
        List of proposed annotations with bbox, class_name, and confidence
        
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If seed_annotations is empty or invalid
    """
    start_time = time.time()
    
    if not seed_annotations or len(seed_annotations) < 2:
        raise ValueError("At least 2 seed annotations are required")
    
    # Load image
    image, display_image_path = await load_cv2_image(image_path)
    logger.info("Loading image from %s", display_image_path)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Load SAM model
    predictor = load_sam_model()
    
    if predictor is None:
        # Fallback: use simple template matching approach
        logger.warning("SAM model not available, using fallback method")
        return await propagate_annotations_fallback(
            image, seed_annotations, similarity_threshold
        )
    
    # Set image in SAM predictor
    predictor.set_image(image_rgb)
    
    # Extract features from seed annotations
    logger.info(f"Extracting features from {len(seed_annotations)} seed annotations")
    seed_features = []
    seed_bboxes = []
    seed_classes = []
    
    for annotation in seed_annotations:
        bbox = annotation["bbox"]
        class_name = annotation.get("class_name", "defect")
        
        features = extract_features_from_bbox(image, bbox)
        seed_features.append(features)
        seed_bboxes.append(bbox)
        seed_classes.append(class_name)
    
    seed_features_array = np.array(seed_features)
    
    # Generate candidate regions using SAM
    # We'll use a grid-based approach to sample points
    logger.info("Generating candidate regions with SAM")
    proposed_annotations = []
    
    # Create a grid of points to sample
    grid_size_x = max(1, min(32, width))
    grid_size_y = max(1, min(32, height))
    step_x = max(1, width // grid_size_x)
    step_y = max(1, height // grid_size_y)
    grid_errors = 0
    logged_grid_errors = 0
    
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            point_x = min(width - 1, i * step_x + step_x // 2)
            point_y = min(height - 1, j * step_y + step_y // 2)
            
            # Skip if point is within existing seed annotations
            is_in_seed = False
            for seed_bbox in seed_bboxes:
                sx, sy, sw, sh = seed_bbox
                if sx <= point_x <= sx + sw and sy <= point_y <= sy + sh:
                    is_in_seed = True
                    break
            
            if is_in_seed:
                continue
            
            # Generate mask for this point using SAM
            try:
                masks, scores, logits = predictor.predict(
                    point_coords=np.array([[point_x, point_y]]),
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                
                # Use the mask with highest score
                best_mask_idx = np.argmax(scores)
                mask = masks[best_mask_idx]
                score = scores[best_mask_idx]
                
                # Convert mask to bounding box
                if mask.sum() == 0:
                    continue
                
                coords = np.argwhere(mask)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                
                # Extract features from proposed region
                proposed_features = extract_features_from_bbox(image, bbox)
                
                # Calculate similarity with seed annotations
                similarities = []
                for seed_feat in seed_features_array:
                    # Cosine similarity
                    dot_product = np.dot(proposed_features, seed_feat)
                    norm_product = np.linalg.norm(proposed_features) * np.linalg.norm(seed_feat)
                    similarity = dot_product / (norm_product + 1e-6)
                    similarities.append(similarity)
                
                max_similarity = max(similarities)
                
                # If similarity exceeds threshold, add to proposals
                if max_similarity >= similarity_threshold:
                    best_seed_idx = np.argmax(similarities)
                    proposed_class = seed_classes[best_seed_idx]
                    
                    proposed_annotations.append({
                        "bbox": bbox,
                        "class_name": proposed_class,
                        "confidence": float(max_similarity),
                        "annotation_id": f"sam_prop_{len(proposed_annotations)}"
                    })
                    
            except Exception as e:
                grid_errors += 1
                if logged_grid_errors < 5:
                    logger.debug("Error processing SAM point (%s, %s): %s", point_x, point_y, e)
                    logged_grid_errors += 1
                continue

    if grid_errors:
        logger.warning(
            "SAM grid sampling encountered %s point-level errors while processing %s",
            grid_errors,
            display_image_path,
        )
    
    processing_time = time.time() - start_time
    logger.info(f"Propagation complete: {len(proposed_annotations)} proposals in {processing_time:.2f}s")
    
    return proposed_annotations


async def propagate_annotations_fallback(
    image: np.ndarray,
    seed_annotations: List[Dict[str, Any]],
    similarity_threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Fallback method for annotation propagation using template matching.
    
    Used when SAM model is not available. Uses OpenCV template matching
    and feature similarity to find regions similar to seed annotations.
    
    Args:
        image: Input image as numpy array
        seed_annotations: List of seed annotations
        similarity_threshold: Minimum similarity for proposals
        
    Returns:
        List of proposed annotations
    """
    logger.info("Using template matching fallback")
    height, width = image.shape[:2]
    proposed_annotations = []
    
    # Extract seed templates
    templates = []
    for annotation in seed_annotations:
        bbox = annotation["bbox"]
        x, y, w, h = [int(v) for v in bbox]
        
        # Ensure valid coordinates
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)
        
        template = image[y:y+h, x:x+w]
        templates.append({
            "image": template,
            "class_name": annotation.get("class_name", "defect"),
            "bbox": bbox
        })
    
    # Use template matching to find similar regions
    for template_info in templates:
        template = template_info["image"]
        class_name = template_info["class_name"]
        
        if template.size == 0:
            continue
        
        # Perform template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        
        # Find locations above threshold
        threshold_adjusted = similarity_threshold * 0.8  # Adjust for template matching
        locations = np.where(result >= threshold_adjusted)
        
        for pt in zip(*locations[::-1]):
            x, y = pt
            w, h = template.shape[1], template.shape[0]
            
            bbox = [float(x), float(y), float(w), float(h)]
            confidence = float(result[y, x])
            
            # Check if this overlaps with existing seed annotations
            is_duplicate = False
            for seed_ann in seed_annotations:
                similarity = calculate_bbox_similarity(bbox, seed_ann["bbox"])
                if similarity > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                proposed_annotations.append({
                    "bbox": bbox,
                    "class_name": class_name,
                    "confidence": confidence,
                    "annotation_id": f"fallback_prop_{len(proposed_annotations)}"
                })
    
    # Remove duplicate proposals
    unique_proposals = []
    for proposal in proposed_annotations:
        is_duplicate = False
        for existing in unique_proposals:
            if calculate_bbox_similarity(proposal["bbox"], existing["bbox"]) > 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_proposals.append(proposal)
    
    logger.info(f"Fallback method found {len(unique_proposals)} proposals")
    return unique_proposals
