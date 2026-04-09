"""
Smart patching and clustering service using CLIP embeddings.

This module provides intelligent patch extraction with AI-suggested sizing
and CLIP-based clustering to group patches into severity categories.
"""

import os
import time
import base64
import logging
from typing import List, Dict, Any, Tuple
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import torch
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Global model registry (shared application state)
from core.state import model_registry
from services.storage_service import load_cv2_image


def load_clip_model() -> Tuple[Any, Any]:
    """
    Load CLIP model and processor for image embedding extraction.
    
    This function lazy-loads the CLIP model from HuggingFace on first use.
    CLIP (Contrastive Language-Image Pre-training) is used to extract
    semantic embeddings from image patches for clustering.
    
    Returns:
        Tuple of (model, processor) for CLIP
        
    Raises:
        RuntimeError: If model loading fails
    """
    if model_registry.get("clip_model") is None:
        logger.info("Loading CLIP model...")
        
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            model_name = "openai/clip-vit-base-patch32"
            
            # Load model and processor
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
        
            # Move to appropriate device
            device = model_registry.get("device", "cpu")
            model.to(device)
            model.eval()
            
            model_registry["clip_model"] = model
            model_registry["clip_processor"] = processor
            
            logger.info(f"CLIP model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"CLIP model loading failed: {e}")
    
    return model_registry["clip_model"], model_registry["clip_processor"]


def calculate_optimal_patch_size(
    annotations: List[Dict[str, Any]],
    padding_factor: float = 1.5,
) -> int:
    """
    Calculate optimal patch size based on annotation bounding boxes.
    
    Uses AI-based analysis of annotation dimensions to suggest the best
    patch size. The algorithm:
    1. Extracts width and height from all annotations
    2. Calculates the median diagonal length
    3. Applies a 1.5x padding factor for context
    4. Rounds to the nearest standard patch size
    
    Args:
        annotations: List of annotations with bbox [x, y, w, h]
        
    Returns:
        Optimal patch size in pixels (multiple of 32, min 64, max 512)
    """
    if not annotations:
        return 224  # Default patch size
    
    # Extract diagonal lengths from all bboxes.
    diagonals = [np.hypot(ann["bbox"][2], ann["bbox"][3]) for ann in annotations]
    
    # Calculate median diagonal
    median_diagonal = np.median(diagonals)
    
    # Apply padding factor for context.
    optimal_size = median_diagonal * padding_factor
    
    # Round to nearest multiple of 32 (standard for neural networks)
    optimal_size = int(np.round(optimal_size / 32) * 32)
    
    # Clamp to reasonable bounds
    optimal_size = max(64, min(512, optimal_size))
    
    logger.info(
        "Calculated optimal patch size=%spx from %s annotations (median diagonal=%.1fpx, padding_factor=%.2f)",
        optimal_size,
        len(annotations),
        median_diagonal,
        padding_factor,
    )
    
    return optimal_size


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert numpy image array to base64 encoded string.
    
    Args:
        image: Image as numpy array (BGR or RGB format)
        
    Returns:
        Base64 encoded string of the image in JPEG format
    """
    # Convert to PIL Image
    if len(image.shape) == 2:
        # Grayscale
        pil_image = Image.fromarray(image)
    else:
        # Color - convert BGR to RGB if needed
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pil_image = Image.fromarray(image_rgb)
    
    # Encode to base64
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str


def base64_to_image(base64_str: str) -> np.ndarray:
    """
    Convert base64 encoded string to numpy image array.
    
    Args:
        base64_str: Base64 encoded image string
        
    Returns:
        Image as numpy array in RGB format
    """
    # Decode base64
    img_data = base64.b64decode(base64_str)
    
    # Convert to PIL Image
    pil_image = Image.open(BytesIO(img_data))
    
    # Convert to numpy array
    image = np.array(pil_image)
    
    return image


async def extract_patches(
    image_path: str,
    annotations: List[Dict[str, Any]],
    patch_size: int = None,
    padding_factor: float = 1.5
) -> Dict[str, Any]:
    """
    Extract image patches from annotated regions with AI-suggested sizing.
    
    This function intelligently extracts patches around annotated defects:
    1. Analyzes annotation dimensions to suggest optimal patch size
    2. Extracts patches with padding for context
    3. Handles edge cases (patches near image boundaries)
    4. Returns patches as base64 encoded images with metadata
    
    **Optimal Patch Size Algorithm:**
    - Calculates median bounding box diagonal across all annotations
    - Multiplies by padding_factor (default 1.5) for context
    - Rounds to nearest 32 pixels for neural network compatibility
    - Clamps to range [64, 512] pixels
    
    Args:
        image_path: Path to the input image file
        annotations: List of annotations with bbox and class_name
        patch_size: Override patch size in pixels (auto-calculated if None)
        padding_factor: Padding multiplier for context (default: 1.5)
        
    Returns:
        Dictionary containing:
            - patches: List of patch metadata with base64 images
            - optimal_patch_size: AI-suggested patch size
            - total_patches: Number of patches extracted
            
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If annotations list is empty
    """
    start_time = time.time()
    
    if not annotations:
        raise ValueError("Annotations list cannot be empty")
    
    image, display_image_path = await load_cv2_image(image_path)

    logger.info("Extracting patches from %s", display_image_path)
    logger.info(f"Annotations: {len(annotations)}, Padding factor: {padding_factor}")
    
    img_height, img_width = image.shape[:2]
    
    # Calculate optimal patch size if not provided
    if patch_size is None:
        patch_size = calculate_optimal_patch_size(
            annotations,
            padding_factor=padding_factor,
        )
    
    logger.info(f"Using patch size: {patch_size}px")
    
    # Extract patches
    patches = []
    
    for idx, annotation in enumerate(annotations):
        bbox = annotation["bbox"]
        class_name = annotation.get("class_name", "defect")
        
        x, y, w, h = bbox
        
        # Calculate center of bounding box
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Calculate patch boundaries (centered on bbox with padding)
        half_patch = patch_size / 2
        
        patch_x1 = int(max(0, center_x - half_patch))
        patch_y1 = int(max(0, center_y - half_patch))
        patch_x2 = int(min(img_width, center_x + half_patch))
        patch_y2 = int(min(img_height, center_y + half_patch))
        
        # Extract patch
        patch_img = image[patch_y1:patch_y2, patch_x1:patch_x2]
        
        # Resize to exact patch_size if needed (due to boundary constraints)
        if patch_img.shape[0] != patch_size or patch_img.shape[1] != patch_size:
            patch_img = cv2.resize(patch_img, (patch_size, patch_size))
        
        # Convert to base64
        patch_base64 = image_to_base64(patch_img)
        
        # Create patch metadata
        patch_metadata = {
            "patch_id": f"patch_{idx}",
            "image_base64": patch_base64,
            "original_bbox": bbox,
            "class_name": class_name,
            "patch_size": patch_size
        }
        
        patches.append(patch_metadata)
    
    processing_time = time.time() - start_time
    logger.info(f"Extracted {len(patches)} patches in {processing_time:.2f}s")
    
    return {
        "patches": patches,
        "optimal_patch_size": patch_size,
        "total_patches": len(patches)
    }


async def cluster_patches(
    patches: List[Dict[str, Any]],
    num_clusters: int = 3
) -> Dict[str, Any]:
    """
    Cluster image patches using CLIP embeddings and K-Means.
    
    This function groups patches into severity categories using:
    1. CLIP visual embeddings for semantic similarity
    2. K-Means clustering to identify natural groupings
    3. Automatic labeling based on defect characteristics
    
    **Clustering Pipeline:**
    1. Extract CLIP embeddings for each patch
    2. Run K-Means clustering (k=3 by default)
    3. Analyze each cluster's characteristics:
       - Average bounding box size (larger = more severe)
       - CLIP embedding centroid (semantic similarity)
    4. Map clusters to severity levels:
       - Severe Defect: Largest average size
       - Minor Defect: Medium average size
       - Clean: Smallest average size
    
    Args:
        patches: List of patch metadata with base64 encoded images
        num_clusters: Number of clusters (default: 3 for Severe/Minor/Clean)
        
    Returns:
        Dictionary containing:
            - severe: List of patches classified as severe defects
            - minor: List of patches classified as minor defects
            - clean: List of patches classified as clean/no defect
            - cluster_stats: Statistics about each cluster
            
    Raises:
        ValueError: If patches list is empty or too small for clustering
        RuntimeError: If CLIP model fails to load or process images
    """
    start_time = time.time()
    
    # Validate inputs
    if not patches:
        raise ValueError("Patches list cannot be empty")
    
    if len(patches) < num_clusters:
        raise ValueError(f"Need at least {num_clusters} patches for clustering, got {len(patches)}")
    
    logger.info(f"Clustering {len(patches)} patches into {num_clusters} groups")
    
    # Load CLIP model
    try:
        model, processor = load_clip_model()
        device = model_registry.get("device", "cpu")
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        raise RuntimeError(f"CLIP model loading failed: {e}")
    
    # Extract CLIP embeddings for each patch
    logger.info("Extracting CLIP embeddings...")
    embeddings: List[np.ndarray | None] = [None] * len(patches)
    decoded_images: List[Image.Image] = []
    decoded_indices: List[int] = []

    for idx, patch in enumerate(patches):
        try:
            # Decode base64 to image
            patch_image = base64_to_image(patch["image_base64"])
            decoded_images.append(Image.fromarray(patch_image))
            decoded_indices.append(idx)
        except Exception as e:
            logger.error(f"Failed to decode patch {patch['patch_id']}: {e}")
            # Use zero embedding as fallback
            embeddings[idx] = np.zeros(512)

    batch_size = 16
    for batch_start in range(0, len(decoded_images), batch_size):
        batch_images = decoded_images[batch_start:batch_start + batch_size]
        batch_indices = decoded_indices[batch_start:batch_start + batch_size]

        try:
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            batch_embeddings = image_features.cpu().numpy()
            for embedding_idx, patch_idx in enumerate(batch_indices):
                embedding = batch_embeddings[embedding_idx].flatten()
                embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
                embeddings[patch_idx] = embedding
        except Exception as e:
            logger.error("Failed to process CLIP batch starting at patch %s: %s", batch_start, e)
            for patch_idx in batch_indices:
                patch_id = patches[patch_idx]["patch_id"]
                logger.error("Using zero embedding fallback for patch %s", patch_id)
                embeddings[patch_idx] = np.zeros(512)

    embeddings_array = np.array([
        embedding if embedding is not None else np.zeros(512)
        for embedding in embeddings
    ])
    logger.info(f"Extracted embeddings shape: {embeddings_array.shape}")
    
    # Run K-Means clustering
    logger.info(f"Running K-Means clustering with k={num_clusters}...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Calculate cluster statistics
    cluster_stats = {}
    cluster_sizes = {}  # Average bounding box size per cluster
    
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_patches = [patches[i] for i in cluster_indices]
        
        # Calculate average bounding box size for this cluster
        if cluster_patches:
            avg_bbox_size = float(np.mean([
                patch["original_bbox"][2] * patch["original_bbox"][3]
                for patch in cluster_patches
            ]))
        else:
            avg_bbox_size = 0.0
        
        cluster_sizes[cluster_id] = avg_bbox_size
        
        cluster_stats[f"cluster_{cluster_id}"] = {
            "num_patches": len(cluster_patches),
            "avg_bbox_area": avg_bbox_size,
            "centroid": kmeans.cluster_centers_[cluster_id].tolist()
        }
    
    # Map clusters to severity levels based on average bbox size
    # Assumption: Larger defects are more severe
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    
    severity_mapping = {}
    if num_clusters >= 3:
        severity_mapping[sorted_clusters[0][0]] = "severe"  # Largest
        severity_mapping[sorted_clusters[-1][0]] = "clean"   # Smallest
        severity_mapping[sorted_clusters[1][0]] = "minor"    # Middle
    elif num_clusters == 2:
        severity_mapping[sorted_clusters[0][0]] = "severe"
        severity_mapping[sorted_clusters[1][0]] = "minor"
    else:
        severity_mapping[0] = "severe"
    
    logger.info(f"Severity mapping: {severity_mapping}")
    
    # Group patches by severity
    severe_patches = []
    minor_patches = []
    clean_patches = []
    
    for idx, label in enumerate(cluster_labels):
        severity = severity_mapping.get(label, "minor")
        
        if severity == "severe":
            severe_patches.append(patches[idx])
        elif severity == "minor":
            minor_patches.append(patches[idx])
        else:
            clean_patches.append(patches[idx])
    
    processing_time = time.time() - start_time
    logger.info(f"Clustering complete in {processing_time:.2f}s")
    logger.info(f"  Severe: {len(severe_patches)}, Minor: {len(minor_patches)}, Clean: {len(clean_patches)}")
    
    return {
        "severe": severe_patches,
        "minor": minor_patches,
        "clean": clean_patches,
        "cluster_stats": cluster_stats
    }
