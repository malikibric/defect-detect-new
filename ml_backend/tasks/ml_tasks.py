"""Background ML tasks with DB status updates and webhook notifications."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx
import redis
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from core.config import settings
from models.db_models import Artifact, Job
from services import synthetic_service
from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

sync_engine = create_engine(settings.DATABASE_SYNC_URL, future=True)
SessionLocal = sessionmaker(bind=sync_engine)
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _update_job_status(
    *,
    job_id: int,
    status: str,
    result_json: dict | None = None,
    error_message: str | None = None,
) -> Job | None:
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.id == job_id))
        if job is None:
            return None
        if job.status == "cancelled" and status in {"running", "succeeded"}:
            return job
        job.status = status
        job.updated_at = _utc_now()
        if result_json is not None:
            job.result_json = json.dumps(result_json)
        if error_message is not None:
            job.error_message = error_message
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def _publish_job_event(job_id: int, status: str, payload: dict | None = None) -> None:
    message = {
        "job_id": job_id,
        "status": status,
        "data": payload or {},
    }
    redis_client.publish(f"jobs:{job_id}", json.dumps(message))


async def _send_webhook(url: str, payload: dict) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(url, json=payload)


@celery_app.task(name="jobs.synthetic.generate")
def synthetic_generate_job(job_id: int) -> dict:
    """Generate synthetic defects in background and persist job result."""
    job = _update_job_status(job_id=job_id, status="running")
    if job is None:
        return {"ok": False, "error": f"Job {job_id} not found"}
    if job.status == "cancelled":
        return {"ok": False, "job_id": job_id, "error": "Job was cancelled"}
    _publish_job_event(job_id, "running")

    try:
        payload = json.loads(job.payload_json or "{}")
        annotation = payload.get("annotation")
        if not annotation:
            raise ValueError("Job payload missing 'annotation'")

        generated_images = asyncio.run(
            synthetic_service.generate_synthetic_defects(
                image_path=payload["image_path"],
                annotation=annotation,
                num_variations=payload.get("num_variations", 10),
                lighting_conditions=payload.get("lighting_conditions"),
                severity_levels=payload.get("severity_levels"),
                output_dir=payload.get("output_dir", "output/synthetic"),
            )
        )

        result = {
            "generated_images": generated_images,
            "total_generated": len(generated_images),
        }

        with SessionLocal() as session:
            db_job = session.scalar(select(Job).where(Job.id == job_id))
            if db_job is not None:
                if db_job.status == "cancelled":
                    return {"ok": False, "job_id": job_id, "error": "Job was cancelled"}
                for index, artifact_uri in enumerate(generated_images):
                    artifact = Artifact(
                        job_id=job_id,
                        artifact_type="synthetic_image",
                        uri=artifact_uri,
                        metadata_json=json.dumps({"index": index}),
                    )
                    session.add(artifact)
                session.commit()

        updated_job = _update_job_status(job_id=job_id, status="succeeded", result_json=result)
        _publish_job_event(job_id, "succeeded", result)

        if updated_job and updated_job.webhook_url:
            webhook_payload = {
                "job_id": updated_job.id,
                "status": updated_job.status,
                "result": result,
            }
            try:
                asyncio.run(_send_webhook(updated_job.webhook_url, webhook_payload))
            except Exception as exc:
                logger.warning("Webhook delivery failed for job %s: %s", updated_job.id, exc)

        return {"ok": True, "job_id": job_id, "result": result}
    except Exception as exc:
        _update_job_status(job_id=job_id, status="failed", error_message=str(exc))
        _publish_job_event(job_id, "failed", {"error": str(exc)})
        logger.exception("Job %s failed", job_id)
        return {"ok": False, "job_id": job_id, "error": str(exc)}
