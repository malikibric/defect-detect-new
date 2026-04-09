"""Redis-backed notification helpers for jobs and websocket clients."""

from __future__ import annotations

import json
import logging
from typing import Any

from redis import asyncio as redis_async

from core.config import settings

logger = logging.getLogger(__name__)


async def publish_job_event(job_id: int, payload: dict[str, Any]) -> None:
    """Publish a job event to Redis Pub/Sub for websocket consumers."""
    channel = f"jobs:{job_id}"
    client = redis_async.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        await client.publish(channel, json.dumps(payload))
    except Exception as exc:
        logger.warning("Failed to publish event for job %s: %s", job_id, exc)
    finally:
        await client.aclose()


def build_job_event(*, job_id: int, status: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create normalized event payloads for websocket updates."""
    return {
        "job_id": job_id,
        "status": status,
        "data": data or {},
    }
