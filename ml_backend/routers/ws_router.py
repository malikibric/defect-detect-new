"""WebSocket endpoints for realtime job status notifications."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from jose import JWTError, jwt
from redis import asyncio as redis_async

from core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/jobs/{job_id}")
async def job_updates(websocket: WebSocket, job_id: int) -> None:
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return

    try:
        jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    redis_client = redis_async.from_url(settings.REDIS_URL, decode_responses=True)
    pubsub = redis_client.pubsub()
    channel = f"jobs:{job_id}"

    await pubsub.subscribe(channel)

    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0)
            if message and message.get("type") == "message":
                payload = message.get("data")
                if isinstance(payload, str):
                    await websocket.send_text(payload)
                else:
                    await websocket.send_text(json.dumps({"job_id": job_id, "data": payload}))
            else:
                await websocket.send_text(json.dumps({"job_id": job_id, "status": "heartbeat"}))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WebSocket error for job %s: %s", job_id, exc)
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()
        await redis_client.aclose()
