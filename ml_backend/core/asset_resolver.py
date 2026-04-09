"""Resolve image references for API requests.

Supports transition from legacy `image_path` to `image_asset_id` and `source_uri`.
"""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.db_models import ImageAsset


async def resolve_image_reference(
    *,
    db: AsyncSession,
    owner_id: int,
    image_path: str | None,
    image_asset_id: int | None,
    source_uri: str | None,
) -> str:
    if image_asset_id is not None:
        result = await db.execute(
            select(ImageAsset).where(ImageAsset.id == image_asset_id, ImageAsset.owner_id == owner_id)
        )
        image_asset = result.scalars().first()
        if image_asset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image asset {image_asset_id} not found",
            )
        return image_asset.source_uri

    if source_uri:
        return source_uri

    if image_path:
        return image_path

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Provide one of image_path, image_asset_id, or source_uri",
    )
