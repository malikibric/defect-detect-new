# FIXED: Added detect_router for frontend upload-and-detect workflow
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
from sqlalchemy import func, select
import torch

from core.config import settings
from routers import sam_router, qa_router, patch_router, synthetic_router, auth_router, jobs_router, ws_router, files_router, project_router, webhook_router, detect_router
from models.database import init_db, close_db
from models.database import AsyncSessionLocal
from models.db_models import Job, User, ImageAsset
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


def _get_allowed_origins() -> list[str]:
    """Return explicitly allowed CORS origins for the backend."""
    return settings.cors_origins_list or [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

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
    logger.info("Initializing Database...")
    await init_db()
        
    logger.info("Starting ML Backend - Loading models...")
    
    # Determine device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model_registry["device"] = device
    
    # Initialize model slots.
    model_registry["sam_model"] = None
    model_registry["sam_predictor"] = None
    model_registry["sam_mask_generator"] = None
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
    await close_db()
    logger.info("ML Backend shutdown complete")


# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Advanced AI capabilities for defect detection and annotation",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
)

# Configure CORS
allowed_origins = _get_allowed_origins()
allow_credentials = "*" not in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router.router, prefix=f"{settings.API_V1_PREFIX}/auth", tags=["Authentication"])
app.include_router(sam_router.router, prefix=f"{settings.API_V1_PREFIX}/sam", tags=["SAM Label Propagation"])
app.include_router(qa_router.router, prefix=f"{settings.API_V1_PREFIX}/qa", tags=["Quality Assurance"])
app.include_router(patch_router.router, prefix=f"{settings.API_V1_PREFIX}/patch", tags=["Smart Patching & Clustering"])
app.include_router(synthetic_router.router, prefix=f"{settings.API_V1_PREFIX}/synthetic", tags=["Synthetic Data Generation"])
app.include_router(jobs_router.router, prefix=f"{settings.API_V1_PREFIX}/jobs", tags=["Jobs"])
app.include_router(files_router.router, prefix=f"{settings.API_V1_PREFIX}/files", tags=["Files"])
app.include_router(project_router.router, prefix=f"{settings.API_V1_PREFIX}/projects", tags=["Projects"])
app.include_router(ws_router.router, prefix=f"{settings.API_V1_PREFIX}/ws", tags=["WebSockets"])
app.include_router(webhook_router.router, prefix=f"{settings.API_V1_PREFIX}/webhooks", tags=["Webhooks"])
app.include_router(detect_router.router, prefix="/api", tags=["Detection (Convenience)"])


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
        "docs": f"{settings.API_V1_PREFIX}/docs"
    }


@app.get(f"{settings.API_V1_PREFIX}/health", tags=["Health"])
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


@app.get(f"{settings.API_V1_PREFIX}/health/liveness", tags=["Health"])
async def liveness_check() -> Dict[str, str]:
    return {"status": "alive"}


@app.get(f"{settings.API_V1_PREFIX}/health/readiness", tags=["Health"])
async def readiness_check() -> Dict[str, Any]:
    startup_status = model_registry.get("startup_status", {})
    has_errors = any(str(value).startswith("error") for value in startup_status.values())
    return {
        "status": "ready" if not has_errors else "degraded",
        "startup_status": startup_status,
    }


@app.get(f"{settings.API_V1_PREFIX}/metrics", tags=["Health"])
async def metrics() -> Dict[str, Any]:
    async with AsyncSessionLocal() as session:
        total_users = await session.scalar(select(func.count(User.id)))
        total_jobs = await session.scalar(select(func.count(Job.id)))
        queued_jobs = await session.scalar(select(func.count(Job.id)).where(Job.status == "queued"))
        running_jobs = await session.scalar(select(func.count(Job.id)).where(Job.status == "running"))
        failed_jobs = await session.scalar(select(func.count(Job.id)).where(Job.status == "failed"))
        total_assets = await session.scalar(select(func.count(ImageAsset.id)))

    return {
        "users_total": int(total_users or 0),
        "jobs_total": int(total_jobs or 0),
        "jobs_queued": int(queued_jobs or 0),
        "jobs_running": int(running_jobs or 0),
        "jobs_failed": int(failed_jobs or 0),
        "assets_total": int(total_assets or 0),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
