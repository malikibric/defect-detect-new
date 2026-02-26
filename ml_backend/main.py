"""
FastAPI application entry point for the ML Backend.

This module initializes the FastAPI application with all routers,
configures lifespan events for model loading, and sets up CORS middleware.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch

from routers import sam_router, qa_router, patch_router, synthetic_router
from services.sam_service import load_sam_model
from services.qa_service import load_yolo_model
from services.patch_service import load_clip_model
from services.synthetic_service import load_diffusion_pipeline
from core.state import model_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for loading ML models at startup.
    
    Models are loaded once during application startup to avoid
    loading overhead on every request. Models are stored in the
    global model_registry for access by service modules.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None: Application runs with loaded models
    """
    logger.info("Starting ML Backend - Loading models...")
    
    # Determine device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model_registry["device"] = device
    
    # Initialize model slots.
    model_registry["sam_model"] = None
    model_registry["sam_predictor"] = None
    model_registry["yolo_model"] = None
    model_registry["clip_model"] = None
    model_registry["clip_processor"] = None
    model_registry["diffusion_pipeline"] = None

    # Load all ML models once at startup as required.
    startup_status: Dict[str, str] = {}

    try:
        load_sam_model()
        startup_status["sam"] = "loaded"
    except Exception as exc:
        startup_status["sam"] = f"error: {exc}"
        logger.exception("Failed to load SAM model during startup")

    try:
        load_yolo_model()
        startup_status["yolo"] = "loaded"
    except Exception as exc:
        startup_status["yolo"] = f"error: {exc}"
        logger.exception("Failed to load YOLO model during startup")

    try:
        load_clip_model()
        startup_status["clip"] = "loaded"
    except Exception as exc:
        startup_status["clip"] = f"error: {exc}"
        logger.exception("Failed to load CLIP model during startup")

    try:
        load_diffusion_pipeline()
        startup_status["diffusion"] = "loaded"
    except Exception as exc:
        startup_status["diffusion"] = f"error: {exc}"
        logger.exception("Failed to load diffusion pipeline during startup")

    model_registry["startup_status"] = startup_status
    
    logger.info("ML Backend startup completed with status: %s", startup_status)
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down ML Backend - Cleaning up models...")
    model_registry.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("ML Backend shutdown complete")


# Initialize FastAPI application
app = FastAPI(
    title="DefectDetect ML Backend",
    description="Advanced AI capabilities for defect detection and annotation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sam_router.router, prefix="/api/sam", tags=["SAM Label Propagation"])
app.include_router(qa_router.router, prefix="/api/qa", tags=["Quality Assurance"])
app.include_router(patch_router.router, prefix="/api/patch", tags=["Smart Patching & Clustering"])
app.include_router(synthetic_router.router, prefix="/api/synthetic", tags=["Synthetic Data Generation"])


@app.get("/", tags=["Health"])
async def root() -> Dict[str, str]:
    """
    Root endpoint for health check.
    
    Returns:
        Dict with status and message
    """
    return {
        "status": "online",
        "message": "DefectDetect ML Backend is running",
        "docs": "/api/docs"
    }


@app.get("/api/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Detailed health check endpoint.
    
    Returns:
        Dict with system information and model status
    """
    return {
        "status": "healthy",
        "device": model_registry.get("device", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "startup_status": model_registry.get("startup_status", {}),
        "models_loaded": {
            "sam": model_registry.get("sam_model") is not None,
            "yolo": model_registry.get("yolo_model") is not None,
            "clip": model_registry.get("clip_model") is not None,
            "diffusion": model_registry.get("diffusion_pipeline") is not None,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
