"""
Synthetic defect generation service using Stable Diffusion inpainting.

This module provides synthetic data augmentation capabilities by generating
realistic variations of defects with different lighting, backgrounds, and severity.
"""

import os
import time
import logging
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

logger = logging.getLogger(__name__)

# Global model registry (shared application state)
from core.state import model_registry


def load_diffusion_pipeline() -> Any:
    """
    Load Stable Diffusion inpainting pipeline for synthetic generation.
    
    This function lazy-loads the diffusion model on first use. The inpainting
    pipeline is used to generate realistic variations of defects by manipulating
    lighting, background, and defect characteristics.
    
    Returns:
        Loaded Stable Diffusion inpainting pipeline
        
    Raises:
        RuntimeError: If pipeline loading fails
    """
    if model_registry.get("diffusion_pipeline") is None:
        logger.info("Loading Stable Diffusion inpainting pipeline...")
        
        try:
            from diffusers import StableDiffusionInpaintPipeline
            
            model_name = "runwayml/stable-diffusion-inpainting"
            
            # Load pipeline
            device = model_registry.get("device", "cpu")
            
            if device == "cuda":
                pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                )
                pipeline = pipeline.to(device)
            else:
                # CPU mode - use float32
                pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
                pipeline = pipeline.to(device)
            
            # Enable memory optimizations
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
            
            model_registry["diffusion_pipeline"] = pipeline
            logger.info(f"Diffusion pipeline loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load diffusion pipeline: {e}")
            raise RuntimeError(f"Diffusion pipeline loading failed: {e}")
    
    return model_registry["diffusion_pipeline"]


def build_prompt(
    defect_type: str,
    lighting: str,
    severity: str = "moderate",
    background: str = "industrial"
) -> str:
    """
    Build a detailed prompt for defect generation with specific conditions.
    
    This function constructs prompts that guide the Stable Diffusion model
    to generate realistic defect variations with controlled characteristics.
    
    **Prompt Engineering Strategy:**
    - Specific defect description with technical terms
    - Lighting condition descriptors
    - Severity modifiers (size, depth, visibility)
    - Background context for realism
    - Negative prompts to avoid artifacts
    
    Args:
        defect_type: Type of defect (e.g., "crack", "corrosion", "scratch")
        lighting: Lighting condition ("dark", "bright", "side-lit", "natural")
        severity: Defect severity level ("minor", "moderate", "severe")
        background: Background context ("industrial", "metal", "concrete", "plastic")
        
    Returns:
        Formatted prompt string for Stable Diffusion
        
    Example:
        >>> prompt = build_prompt("crack", "bright", "severe", "concrete")
        >>> print(prompt)
        "A severe surface crack defect on concrete material, bright lighting,
        high detail, photorealistic, industrial quality inspection, sharp focus"
    """
    # Severity descriptors
    severity_map = {
        "minor": "small, shallow, barely visible",
        "moderate": "medium-sized, clearly visible",
        "severe": "large, deep, extensive"
    }
    
    severity_desc = severity_map.get(severity, "medium-sized")
    
    # Lighting descriptors
    lighting_map = {
        "dark": "dim lighting, shadows, low light conditions",
        "bright": "bright lighting, well-lit, high exposure",
        "side-lit": "side lighting, directional light, high contrast",
        "natural": "natural daylight, soft lighting, balanced exposure",
        "overhead": "overhead lighting, fluorescent, industrial lighting"
    }
    
    lighting_desc = lighting_map.get(lighting, "natural daylight")
    
    # Construct prompt
    prompt = (
        f"A {severity_desc} {defect_type} defect on {background} surface, "
        f"{lighting_desc}, photorealistic, high detail, sharp focus, "
        f"industrial quality inspection, professional photography, "
        f"8k resolution, macro photography"
    )
    
    return prompt


def build_negative_prompt() -> str:
    """
    Build negative prompt to avoid common artifacts in generated images.
    
    Returns:
        Negative prompt string listing undesired characteristics
    """
    return (
        "blurry, low quality, distorted, cartoon, painting, drawing, "
        "artistic, unrealistic, oversaturated, text, watermark, "
        "multiple defects, unclear, noisy, grainy"
    )


async def generate_synthetic_defects(
    image_path: str,
    annotation: Dict[str, Any],
    num_variations: int = 10,
    lighting_conditions: List[str] = None,
    severity_levels: List[str] = None,
    output_dir: str = "output/synthetic"
) -> List[str]:
    """
    Generate synthetic defect variations using Stable Diffusion inpainting.
    
    This function creates realistic variations of a defect annotation by:
    1. Extracting the defect region from the original image
    2. Creating an inpainting mask from the annotation bounding box
    3. Using Stable Diffusion to generate variations with different:
       - Lighting conditions (dark, bright, side-lit, natural)
       - Defect severity levels (minor, moderate, severe)
       - Background textures (maintaining material context)
    
    **Inpainting Pipeline Flow:**
    
    1. **Image Preparation**:
       - Load source image
       - Extract defect region based on bounding box
       - Create binary mask for inpainting area
    
    2. **Prompt Engineering**:
       - Build detailed prompts for each variation
       - Combine defect type + lighting + severity
       - Add negative prompts to avoid artifacts
    
    3. **Generation**:
       - Run Stable Diffusion inpainting for each variation
       - Control randomness with different seeds
       - Apply guidance scale for prompt adherence
    
    4. **Post-processing**:
       - Save generated variations to output directory
       - Maintain original image dimensions
       - Preserve file naming convention
    
    Args:
        image_path: Path to the source image file
        annotation: Annotation dict with bbox and class_name defining the defect
        num_variations: Number of synthetic variations to generate (default: 10)
        lighting_conditions: List of lighting conditions to apply (default: ["dark", "bright", "side-lit"])
        severity_levels: List of severity levels (default: ["minor", "moderate", "severe"])
        output_dir: Directory to save generated images (default: "output/synthetic")
        
    Returns:
        List of file paths to generated synthetic images
        
    Raises:
        FileNotFoundError: If source image does not exist
        RuntimeError: If diffusion pipeline fails
        ValueError: If annotation is invalid
        
    Example:
        >>> annotation = {"bbox": [100, 100, 50, 50], "class_name": "crack"}
        >>> paths = await generate_synthetic_defects(
        ...     "defect.jpg",
        ...     annotation,
        ...     num_variations=15
        ... )
        >>> print(f"Generated {len(paths)} variations")
    """
    start_time = time.time()
    
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not annotation or "bbox" not in annotation:
        raise ValueError("Invalid annotation: must contain 'bbox' field")
    
    # Set defaults
    if lighting_conditions is None:
        lighting_conditions = ["dark", "bright", "side-lit"]
    
    if severity_levels is None:
        severity_levels = ["minor", "moderate", "severe"]
    
    logger.info(f"Generating {num_variations} synthetic variations from {image_path}")
    logger.info(f"Lighting conditions: {lighting_conditions}")
    logger.info(f"Severity levels: {severity_levels}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    img_height, img_width = image_array.shape[:2]
    
    # Extract annotation info
    bbox = annotation["bbox"]
    class_name = annotation.get("class_name", "defect")
    x, y, w, h = [int(v) for v in bbox]
    
    # Ensure bbox is within image bounds
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
    logger.info(f"Defect region: x={x}, y={y}, w={w}, h={h}, class={class_name}")
    
    # Create mask for inpainting
    # Mask should be white (255) where we want to inpaint, black (0) elsewhere
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Expand the mask slightly beyond the bbox for better blending
    expansion = int(max(w, h) * 0.1)  # 10% expansion
    mask_x1 = max(0, x - expansion)
    mask_y1 = max(0, y - expansion)
    mask_x2 = min(img_width, x + w + expansion)
    mask_y2 = min(img_height, y + h + expansion)
    
    mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
    mask_image = Image.fromarray(mask).convert("L")
    
    # Load diffusion pipeline
    try:
        pipeline = load_diffusion_pipeline()
    except Exception as e:
        logger.error(f"Failed to load diffusion pipeline: {e}")
        # Fallback: create simple variations using traditional augmentation
        return await generate_synthetic_defects_fallback(
            image_path, annotation, num_variations, output_dir
        )
    
    # Generate variations
    generated_paths = []
    base_filename = Path(image_path).stem
    
    negative_prompt = build_negative_prompt()
    
    for i in range(num_variations):
        # Cycle through lighting and severity combinations
        lighting = lighting_conditions[i % len(lighting_conditions)]
        severity = severity_levels[i % len(severity_levels)]
        
        # Build prompt
        prompt = build_prompt(class_name, lighting, severity)
        
        logger.info(f"Generating variation {i+1}/{num_variations}: {lighting}, {severity}")
        logger.debug(f"Prompt: {prompt}")
        
        try:
            # Generate image using Stable Diffusion inpainting
            # Pipeline: image + mask + prompt → inpainted_image
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    mask_image=mask_image,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device=pipeline.device).manual_seed(42 + i)
                )
            
            generated_image = result.images[0]
            
            # Save generated image
            output_filename = f"{base_filename}_synthetic_{i:03d}_{lighting}_{severity}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            generated_image.save(output_path)
            generated_paths.append(output_path)
            
            logger.debug(f"Saved variation to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate variation {i}: {e}")
            continue
    
    processing_time = time.time() - start_time
    logger.info(f"Generated {len(generated_paths)} variations in {processing_time:.2f}s")
    logger.info(f"Output directory: {output_dir}")
    
    return generated_paths


async def generate_synthetic_defects_fallback(
    image_path: str,
    annotation: Dict[str, Any],
    num_variations: int = 10,
    output_dir: str = "output/synthetic"
) -> List[str]:
    """
    Fallback method for synthetic generation using traditional augmentation.
    
    Used when Stable Diffusion is not available. Applies classical image
    processing techniques to create variations.
    
    Args:
        image_path: Path to source image
        annotation: Annotation with defect region
        num_variations: Number of variations to generate
        output_dir: Output directory
        
    Returns:
        List of generated image paths
    """
    logger.info("Using fallback augmentation method")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    bbox = annotation["bbox"]
    x, y, w, h = [int(v) for v in bbox]
    
    generated_paths = []
    base_filename = Path(image_path).stem
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_variations):
        # Create variation using traditional augmentation
        augmented = image.copy()
        
        # Apply different transformations
        variation_type = i % 4
        
        if variation_type == 0:
            # Brightness adjustment
            alpha = 0.7 + (i % 3) * 0.3  # 0.7, 1.0, 1.3
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=0)
        
        elif variation_type == 1:
            # Contrast adjustment
            alpha = 0.8 + (i % 3) * 0.2  # 0.8, 1.0, 1.2
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=10)
        
        elif variation_type == 2:
            # Gaussian blur for different focus
            kernel_size = 3 + (i % 2) * 2  # 3 or 5
            augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
        
        else:
            # Add slight noise
            noise = np.random.randint(-20, 20, augmented.shape, dtype=np.int16)
            augmented = np.clip(augmented.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save variation
        output_filename = f"{base_filename}_synthetic_{i:03d}_fallback.png"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, augmented)
        generated_paths.append(output_path)
    
    logger.info(f"Generated {len(generated_paths)} fallback variations")
    return generated_paths
