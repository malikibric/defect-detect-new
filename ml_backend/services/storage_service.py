"""Storage abstraction for handling uploaded assets.

Current implementation is local-disk based and can be swapped later with S3/GCS.
"""

from __future__ import annotations

import hashlib
import uuid
from io import BytesIO
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

import cv2
import httpx
import numpy as np
from PIL import Image

from fastapi import UploadFile

from core.config import settings
from core.path_security import resolve_existing_input_path, safe_display_path


class StorageService(Protocol):
    async def save_upload(self, file: UploadFile, max_size_bytes: int | None = None) -> dict[str, str]:
        ...


class LocalStorageService:
    """Store uploaded files under a local root directory."""

    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root or settings.LOCAL_STORAGE_ROOT).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile, max_size_bytes: int | None = None) -> dict[str, str]:
        suffix = Path(file.filename or "upload.bin").suffix
        filename = f"{uuid.uuid4().hex}{suffix}"
        destination = self.root / filename

        sha256 = hashlib.sha256()
        size_bytes = 0

        try:
            with destination.open("wb") as output:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if max_size_bytes is not None and size_bytes > max_size_bytes:
                        raise ValueError(f"Upload exceeds maximum allowed size of {max_size_bytes} bytes")
                    sha256.update(chunk)
                    output.write(chunk)
        except ValueError:
            destination.unlink(missing_ok=True)
            raise

        await file.close()

        return {
            "uri": destination.as_uri(),
            "checksum": sha256.hexdigest(),
            "size_bytes": str(size_bytes),
            "filename": file.filename or filename,
            "content_type": file.content_type or "application/octet-stream",
        }


class S3StorageService:
    """Placeholder for S3-compatible uploads (implemented in next infra phase)."""

    async def save_upload(self, file: UploadFile) -> dict[str, str]:
        raise NotImplementedError("S3 storage backend is not implemented yet")


def get_storage_service() -> StorageService:
    backend = settings.STORAGE_BACKEND.lower().strip()
    if backend == "local":
        return LocalStorageService()
    if backend == "s3":
        return S3StorageService()
    raise ValueError(f"Unsupported STORAGE_BACKEND: {settings.STORAGE_BACKEND}")


def _is_remote_uri(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _is_file_uri(value: str) -> bool:
    return value.startswith("file://")


def _display_reference(value: str) -> str:
    if _is_remote_uri(value):
        return value
    if _is_file_uri(value):
        parsed = urlparse(value)
        return Path(unquote(parsed.path)).name or value
    return safe_display_path(value)


async def read_image_bytes(reference: str) -> tuple[bytes, str]:
    """Read image bytes from local path, file URI, or HTTP(S) URI."""
    if _is_remote_uri(reference):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(reference)
            response.raise_for_status()
            return response.content, _display_reference(reference)

    if _is_file_uri(reference):
        parsed = urlparse(reference)
        file_path = unquote(parsed.path)
        resolved = resolve_existing_input_path(file_path)
        return resolved.read_bytes(), safe_display_path(resolved)

    resolved = resolve_existing_input_path(reference)
    return resolved.read_bytes(), safe_display_path(resolved)


async def load_cv2_image(reference: str) -> tuple[np.ndarray, str]:
    """Load an image reference into an OpenCV image array (BGR)."""
    image_bytes, display_reference = await read_image_bytes(reference)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {display_reference}")
    return image, display_reference


async def load_pil_image_rgb(reference: str) -> tuple[Image.Image, str]:
    """Load an image reference into a PIL RGB image."""
    image_bytes, display_reference = await read_image_bytes(reference)
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to decode image: {display_reference}") from exc
    return image, display_reference


def image_reference_stem(reference: str) -> str:
    """Return a stable stem from path/URI for generated artifact naming."""
    if _is_remote_uri(reference):
        parsed = urlparse(reference)
        name = Path(parsed.path).stem
        return name or "image"
    if _is_file_uri(reference):
        parsed = urlparse(reference)
        name = Path(unquote(parsed.path)).stem
        return name or "image"
    return Path(reference).stem or "image"
