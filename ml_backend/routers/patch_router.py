"""
Smart patching and clustering router for patch extraction and analysis.

This module exposes REST API endpoints for extracting patches from images
and clustering them by severity using CLIP embeddings.
"""

import time
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from models.schemas import (
    PatchExtractRequest,
    PatchExtractResponse,
    ClusterPatchesRequest,
    ClusterPatchesResponse,
    PatchMetadata
)

from services import patch_service
from core.deps import get_current_active_user
from core.asset_resolver import resolve_image_reference
from models.db_models import User
from models.database import get_db
from core.path_security import safe_display_path
from core.serialization import patches_from_dicts, patches_to_dicts

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_patch_metadata_list(payloads: list[dict]) -> list[PatchMetadata]:
    """Convert patch payload dictionaries into response models."""
    return patches_from_dicts(payloads)


@router.post(
    "/extract",
    response_model=PatchExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract patches from annotated regions",
    description="""
    Extract image patches from annotated regions with AI-suggested optimal sizing.
    
    This endpoint intelligently extracts patches around defect annotations,
    automatically determining the best patch size if not specified.
    
    **How it works:**
    1. Analyzes annotation bounding box dimensions
    2. Calculates optimal patch size using median diagonal × 1.5
    3. Extracts patches centered on annotations with padding for context
    4. Returns patches as base64-encoded images with metadata
    
    **Optimal Patch Size Algorithm:**
    - Examines all annotation bounding boxes
    - Calculates median diagonal length (√(w² + h²))
    - Applies padding factor (default 1.5x) for context
    - Rounds to nearest 32 pixels (neural network friendly)
    - Clamps to range [64, 512] pixels
    
    **Use cases:**
    - Preparing patches for ML model training
    - Creating defect thumbnails for review
    - Extracting regions for detailed analysis
    - Building patch-based datasets
    
    **Parameters:**
    - `image_path`: Path to the image file on the server
    - `annotations`: List of annotations to extract patches from
    - `patch_size`: Override patch size (auto-calculated if null)
    - `padding_factor`: Context padding multiplier (default: 1.5)
    
    **Returns:**
    - `patches`: List of patches with base64 images and metadata
    - `optimal_patch_size`: AI-suggested patch size used
    - `total_patches`: Number of patches extracted
    """
)
async def extract_patches(
    request: PatchExtractRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> PatchExtractResponse:
    """
    Extract patches from annotated regions with optimal sizing.
    
    Args:
        request: PatchExtractRequest with image path and annotations
        
    Returns:
        PatchExtractResponse with extracted patches
        
    Raises:
        HTTPException 404: If image file is not found
        HTTPException 422: If annotations are invalid
        HTTPException 500: If extraction fails
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
            "User %s extracting patches from %s",
            current_user.id,
            safe_display_path(resolved_image_path),
        )
        logger.info(f"Annotations: {len(request.annotations)}, Patch size: {request.patch_size}")
        
        # Call service
        result = await patch_service.extract_patches(
            image_path=resolved_image_path,
            annotations=[annotation.model_dump(exclude_none=True) for annotation in request.annotations],
            patch_size=request.patch_size,
            padding_factor=request.padding_factor
        )
        
        patches = _build_patch_metadata_list(result["patches"])
        
        processing_time = time.time() - start_time
        
        logger.info(f"Extraction successful: {len(patches)} patches in {processing_time:.2f}s")
        
        return PatchExtractResponse(
            patches=patches,
            optimal_patch_size=result["optimal_patch_size"],
            total_patches=result["total_patches"]
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
        logger.error(f"Patch extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Patch extraction failed: {str(e)}"
        )


@router.post(
    "/cluster",
    response_model=ClusterPatchesResponse,
    status_code=status.HTTP_200_OK,
    summary="Cluster patches by severity using CLIP",
    description="""
    Cluster image patches into severity categories using CLIP embeddings and K-Means.
    
    This endpoint uses deep learning to automatically group patches into
    severity levels: Severe Defect, Minor Defect, and Clean.
    
    **How it works:**
    1. Extracts CLIP visual embeddings from each patch
       - CLIP provides semantic understanding of image content
       - Embeddings capture defect characteristics and severity
    2. Runs K-Means clustering on embeddings
       - Groups patches with similar visual characteristics
       - Default k=3 for three severity levels
    3. Maps clusters to severity levels
       - Analyzes average bounding box size per cluster
       - Larger defects → Severe, smaller → Minor/Clean
    4. Returns patches grouped by category
    
    **Severity Mapping Logic:**
    - **Severe Defect**: Cluster with largest average defect size
    - **Minor Defect**: Cluster with medium average defect size
    - **Clean**: Cluster with smallest average defect size
    
    **Use cases:**
    - Automatically prioritize defects by severity
    - Filter training data by defect severity
    - Create balanced datasets across severity levels
    - Quality control and defect triage
    
    **Parameters:**
    - `patches`: List of patch metadata from /patch/extract
    - `num_clusters`: Number of clusters (default: 3)
    
    **Returns:**
    - `severe`: Patches classified as severe defects
    - `minor`: Patches classified as minor defects
    - `clean`: Patches classified as clean/no defect
    - `cluster_stats`: Statistics for each cluster
    """
)
async def cluster_patches(
    request: ClusterPatchesRequest,
    current_user: User = Depends(get_current_active_user),
) -> ClusterPatchesResponse:
    """
    Cluster patches by severity using CLIP and K-Means.
    
    Args:
        request: ClusterPatchesRequest with patches to cluster
        
    Returns:
        ClusterPatchesResponse with patches grouped by severity
        
    Raises:
        HTTPException 422: If patches are invalid or insufficient
        HTTPException 500: If clustering fails
    """
    start_time = time.time()
    
    try:
        logger.info(
            "User %s clustering %s patches into %s groups",
            current_user.id,
            len(request.patches),
            request.num_clusters,
        )
        
        # Call service
        result = await patch_service.cluster_patches(
            patches=patches_to_dicts(request.patches),
            num_clusters=request.num_clusters
        )
        
        severe = _build_patch_metadata_list(result["severe"])
        minor = _build_patch_metadata_list(result["minor"])
        clean = _build_patch_metadata_list(result["clean"])
        
        processing_time = time.time() - start_time
        
        logger.info(f"Clustering successful in {processing_time:.2f}s")
        logger.info(f"  Severe: {len(severe)}, Minor: {len(minor)}, Clean: {len(clean)}")
        
        return ClusterPatchesResponse(
            severe=severe,
            minor=minor,
            clean=clean,
            cluster_stats=result["cluster_stats"]
        )
        
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input: {str(e)}"
        )
    
    except RuntimeError as e:
        logger.error(f"Clustering failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clustering failed: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in clustering: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clustering failed: {str(e)}"
        )
