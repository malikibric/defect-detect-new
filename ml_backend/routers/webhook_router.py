"""Webhook endpoint for external callback integrations (test/ops)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/job-events")
async def receive_job_event(request: Request) -> dict[str, Any]:
    payload = await request.json()
    logger.info("Received job webhook payload: %s", payload)
    return {"ok": True}
