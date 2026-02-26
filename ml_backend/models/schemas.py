"""
Pydantic models for request/response schemas across all ML services.

This module defines the data structures used for API communication,
including annotation formats, QA reports, patch metadata, and synthetic data requests.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """COCO-format bounding box [x, y, width, height]."""
    
    x: float = Field(..., description="Top-left x coordinate")
    y: float = Field(..., description="Top-left y coordinate")
    width: float = Field(..., ge=0, description="Bounding box width")
    height: float = Field(..., ge=0, description="Bounding box height")


class Annotation(BaseModel):
    """Single annotation with bounding box and metadata."""
    
    bbox: List[float] = Field(..., description="Bounding box in [x, y, w, h] format")
    class_name: str = Field(..., description="Defect class label")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    annotation_id: Optional[str] = Field(None, description="Unique annotation identifier")
    
    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        """Ensure bbox has exactly 4 values [x, y, w, h]."""
        if len(v) != 4:
            raise ValueError("Bounding box must have exactly 4 values [x, y, w, h]")
        if v[2] <= 0 or v[3] <= 0:
            raise ValueError("Width and height must be positive")
        return v


class SAMPropagateRequest(BaseModel):
    """Request schema for SAM label propagation."""
    
    image_path: str = Field(..., description="Path to the input image file")
    seed_annotations: List[Annotation] = Field(
        ..., 
        min_items=2, 
        max_items=10,
        description="Seed annotations for propagation (2-10 examples)"
    )
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for proposing new annotations"
    )


class SAMPropagateResponse(BaseModel):
    """Response schema for SAM label propagation."""
    
    proposed_annotations: List[Annotation] = Field(..., description="AI-proposed annotations")
    total_proposed: int = Field(..., description="Total number of proposals")
    processing_time_seconds: float = Field(..., description="Time taken for processing")


class QACheckRequest(BaseModel):
    """Request schema for QA validation."""
    
    image_path: str = Field(..., description="Path to the input image file")
    human_annotations: List[Annotation] = Field(..., description="Human-provided annotations to validate")
    iou_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="IoU threshold for matching predictions"
    )


class QACheckResponse(BaseModel):
    """Response schema for QA validation results."""
    
    missed_defects: List[Annotation] = Field(
        ..., 
        description="YOLO detections not covered by human annotations"
    )
    size_warnings: List[Dict[str, Any]] = Field(
        ...,
        description="Annotations with size deviations >40% from median"
    )
    confirmed: List[Annotation] = Field(
        ...,
        description="Annotations matching YOLO predictions (IoU > threshold)"
    )
    total_human_annotations: int = Field(..., description="Total human annotations provided")
    total_ai_detections: int = Field(..., description="Total AI detections found")
    processing_time_seconds: float = Field(..., description="Time taken for QA check")


class PatchExtractRequest(BaseModel):
    """Request schema for patch extraction."""
    
    image_path: str = Field(..., description="Path to the input image file")
    annotations: List[Annotation] = Field(..., description="Annotations to extract patches from")
    patch_size: Optional[int] = Field(
        None,
        ge=64,
        le=1024,
        description="Patch size in pixels (auto-calculated if None)"
    )
    padding_factor: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="Padding multiplier for patch extraction"
    )


class PatchMetadata(BaseModel):
    """Metadata for an extracted image patch."""
    
    patch_id: str = Field(..., description="Unique patch identifier")
    image_base64: str = Field(..., description="Base64-encoded patch image")
    original_bbox: List[float] = Field(..., description="Original bounding box")
    class_name: str = Field(..., description="Defect class")
    patch_size: int = Field(..., description="Actual patch size used")


class PatchExtractResponse(BaseModel):
    """Response schema for patch extraction."""
    
    patches: List[PatchMetadata] = Field(..., description="Extracted patches with metadata")
    optimal_patch_size: int = Field(..., description="AI-suggested optimal patch size")
    total_patches: int = Field(..., description="Total number of patches extracted")


class ClusterPatchesRequest(BaseModel):
    """Request schema for patch clustering."""
    
    patches: List[PatchMetadata] = Field(..., description="Patches to cluster")
    num_clusters: int = Field(default=3, ge=2, le=10, description="Number of clusters")


class ClusterPatchesResponse(BaseModel):
    """Response schema for patch clustering."""
    
    severe: List[PatchMetadata] = Field(..., description="Severe defect patches")
    minor: List[PatchMetadata] = Field(..., description="Minor defect patches")
    clean: List[PatchMetadata] = Field(..., description="Clean/no defect patches")
    cluster_stats: Dict[str, Any] = Field(..., description="Clustering statistics")


class SyntheticGenerateRequest(BaseModel):
    """Request schema for synthetic defect generation."""
    
    image_path: str = Field(..., description="Path to the source image")
    annotation: Annotation = Field(..., description="Defect annotation to vary")
    num_variations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of synthetic variations to generate"
    )
    lighting_conditions: List[str] = Field(
        default=["dark", "bright", "side-lit"],
        description="Lighting variations to apply"
    )
    severity_levels: List[str] = Field(
        default=["minor", "moderate", "severe"],
        description="Defect severity levels"
    )


class SyntheticGenerateResponse(BaseModel):
    """Response schema for synthetic defect generation."""
    
    generated_images: List[str] = Field(..., description="Paths to generated synthetic images")
    total_generated: int = Field(..., description="Total images generated")
    output_directory: str = Field(..., description="Directory containing generated images")
    processing_time_seconds: float = Field(..., description="Time taken for generation")
