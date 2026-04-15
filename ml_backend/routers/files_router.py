"""File upload and asset registry endpoints."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.deps import get_current_active_user
from models.database import get_db
from models.db_models import ImageAsset, Project, User
from models.schemas import FileUploadResponse
from services.storage_service import get_storage_service

router = APIRouter()
logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    project_id: int | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> FileUploadResponse:
    logger.info("User %s uploading file %s", current_user.id, file.filename)
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    storage = get_storage_service()
    try:
        metadata = await storage.save_upload(file, max_size_bytes=max_size_bytes)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds max size of {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    if project_id is not None:
        result = await db.execute(select(Project).where(Project.id == project_id, Project.owner_id == current_user.id))
        project = result.scalars().first()
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    image_asset = ImageAsset(
        owner_id=current_user.id,
        project_id=project_id,
        source_uri=metadata["uri"],
        content_type=metadata["content_type"],
        checksum=metadata["checksum"],
        original_filename=metadata["filename"],
    )

    db.add(image_asset)
    await db.commit()
    await db.refresh(image_asset)

    return FileUploadResponse(
        id=image_asset.id,
        source_uri=image_asset.source_uri,
        checksum=image_asset.checksum,
        content_type=image_asset.content_type,
        original_filename=image_asset.original_filename,
        project_id=image_asset.project_id,
        created_at=image_asset.created_at.isoformat(),
    )


@router.get("/", response_model=list[FileUploadResponse])
async def list_assets(
    project_id: int | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> list[FileUploadResponse]:
    query = select(ImageAsset).where(ImageAsset.owner_id == current_user.id)
    if project_id is not None:
        query = query.where(ImageAsset.project_id == project_id)
    query = query.order_by(ImageAsset.id.desc())

    result = await db.execute(query)
    assets = result.scalars().all()
    return [
        FileUploadResponse(
            id=asset.id,
            source_uri=asset.source_uri,
            checksum=asset.checksum,
            content_type=asset.content_type,
            original_filename=asset.original_filename,
            project_id=asset.project_id,
            created_at=asset.created_at.isoformat(),
        )
        for asset in assets
    ]


@router.get("/{asset_id}", response_model=FileUploadResponse)
async def get_asset(
    asset_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> FileUploadResponse:
    result = await db.execute(select(ImageAsset).where(ImageAsset.id == asset_id, ImageAsset.owner_id == current_user.id))
    asset = result.scalars().first()
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")

    return FileUploadResponse(
        id=asset.id,
        source_uri=asset.source_uri,
        checksum=asset.checksum,
        content_type=asset.content_type,
        original_filename=asset.original_filename,
        project_id=asset.project_id,
        created_at=asset.created_at.isoformat(),
    )
