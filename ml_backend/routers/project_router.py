"""Project/workspace management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.deps import get_current_active_user
from models.database import get_db
from models.db_models import Project, User
from models.schemas import ProjectCreateRequest, ProjectResponse

router = APIRouter()


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    request: ProjectCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> ProjectResponse:
    project = Project(
        name=request.name,
        description=request.description,
        owner_id=current_user.id,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        owner_id=project.owner_id,
        created_at=project.created_at.isoformat(),
    )


@router.get("/", response_model=list[ProjectResponse])
async def list_projects(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> list[ProjectResponse]:
    result = await db.execute(select(Project).where(Project.owner_id == current_user.id).order_by(Project.id.desc()))
    projects = result.scalars().all()
    return [
        ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            owner_id=project.owner_id,
            created_at=project.created_at.isoformat(),
        )
        for project in projects
    ]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> ProjectResponse:
    result = await db.execute(select(Project).where(Project.id == project_id, Project.owner_id == current_user.id))
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        owner_id=project.owner_id,
        created_at=project.created_at.isoformat(),
    )
