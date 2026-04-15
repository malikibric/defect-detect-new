"""Job management endpoints for asynchronous background ML workloads."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.deps import get_current_active_user
from core.asset_resolver import resolve_image_reference
from core.notifications import build_job_event, publish_job_event
from models.database import get_db
from models.db_models import Artifact, ImageAsset, Job, Project, User
from models.schemas import ArtifactResponse, JobCreateRequest, JobDetailResponse, JobResponse
from tasks.celery_app import celery_app
from tasks.ml_tasks import synthetic_generate_job

router = APIRouter()


@router.get("/", response_model=list[JobResponse])
async def list_jobs(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> list[JobResponse]:
    result = await db.execute(select(Job).where(Job.created_by_id == current_user.id).order_by(Job.id.desc()))
    jobs = result.scalars().all()
    return [
        JobResponse(
            id=job.id,
            type=job.type,
            status=job.status,
            celery_task_id=job.celery_task_id,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            error_message=job.error_message,
        )
        for job in jobs
    ]


@router.post("/", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: JobCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> JobResponse:
    if request.job_type != "synthetic.generate":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported job_type. Currently supported: synthetic.generate",
        )

    linked_project_id = None
    linked_image_asset_id = request.image_asset_id

    if request.image_asset_id is not None:
        asset_result = await db.execute(select(ImageAsset).where(ImageAsset.id == request.image_asset_id))
        asset = asset_result.scalars().first()
        if asset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image asset not found")
        if asset.project_id is not None:
            project_result = await db.execute(
                select(Project).where(Project.id == asset.project_id, Project.owner_id == current_user.id)
            )
            project = project_result.scalars().first()
            if project is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image asset not found")
            linked_project_id = project.id

    resolved_image_path = await resolve_image_reference(
        db=db,
        owner_id=current_user.id,
        image_path=request.image_path,
        image_asset_id=request.image_asset_id,
        source_uri=request.source_uri,
    )

    payload = {
        **request.payload,
        "image_path": resolved_image_path,
    }

    job = Job(
        type=request.job_type,
        status="queued",
        payload_json=json.dumps(payload),
        webhook_url=request.webhook_url,
        project_id=linked_project_id,
        image_asset_id=linked_image_asset_id,
        created_by_id=current_user.id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    task = synthetic_generate_job.delay(job.id)
    job.celery_task_id = task.id
    await db.commit()
    await db.refresh(job)
    await publish_job_event(job.id, build_job_event(job_id=job.id, status="queued", data={"task_id": task.id}))

    return JobResponse(
        id=job.id,
        type=job.type,
        status=job.status,
        celery_task_id=job.celery_task_id,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        error_message=job.error_message,
    )


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> JobDetailResponse:
    result = await db.execute(select(Job).where(Job.id == job_id, Job.created_by_id == current_user.id))
    job = result.scalars().first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    artifacts_result = await db.execute(select(Artifact).where(Artifact.job_id == job.id).order_by(Artifact.id.asc()))
    artifacts = artifacts_result.scalars().all()

    parsed_result_json = None
    if job.result_json:
        try:
            parsed_result_json = json.loads(job.result_json)
        except Exception:
            parsed_result_json = {"raw": job.result_json}

    return JobDetailResponse(
        id=job.id,
        type=job.type,
        status=job.status,
        celery_task_id=job.celery_task_id,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        error_message=job.error_message,
        result_json=parsed_result_json,
        artifacts=[
            ArtifactResponse(
                id=artifact.id,
                artifact_type=artifact.artifact_type,
                uri=artifact.uri,
                metadata_json=artifact.metadata_json,
                created_at=artifact.created_at.isoformat(),
            )
            for artifact in artifacts
        ],
    )


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> JobResponse:
    result = await db.execute(select(Job).where(Job.id == job_id, Job.created_by_id == current_user.id).with_for_update())
    job = result.scalars().first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.status in {"succeeded", "failed", "cancelled"}:
        return JobResponse(
            id=job.id,
            type=job.type,
            status=job.status,
            celery_task_id=job.celery_task_id,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            error_message=job.error_message,
        )

    if job.celery_task_id:
        celery_app.control.revoke(job.celery_task_id, terminate=True)

    job.status = "cancelled"
    await db.commit()
    await db.refresh(job)

    await publish_job_event(job.id, build_job_event(job_id=job.id, status="cancelled"))

    return JobResponse(
        id=job.id,
        type=job.type,
        status=job.status,
        celery_task_id=job.celery_task_id,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        error_message=job.error_message,
    )
