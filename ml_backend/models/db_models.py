from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from models.database import Base


def _utcnow() -> datetime:
    """Return current UTC time as a naive datetime (strips tzinfo for DB compatibility)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_utcnow)

    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    image_assets = relationship("ImageAsset", back_populates="owner", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="created_by", cascade="all, delete-orphan")


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=_utcnow)

    owner = relationship("User", back_populates="projects")
    image_assets = relationship("ImageAsset", back_populates="project", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="project", cascade="all, delete-orphan")


class ImageAsset(Base):
    __tablename__ = "image_assets"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True)
    source_uri = Column(String(1024), nullable=False)
    content_type = Column(String(255), nullable=True)
    checksum = Column(String(128), nullable=True)
    original_filename = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    owner = relationship("User", back_populates="image_assets")
    project = relationship("Project", back_populates="image_assets")
    jobs = relationship("Job", back_populates="image_asset")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(100), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="queued", index=True)
    payload_json = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    celery_task_id = Column(String(255), nullable=True, index=True)
    webhook_url = Column(String(1024), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True)
    image_asset_id = Column(Integer, ForeignKey("image_assets.id", ondelete="SET NULL"), nullable=True, index=True)
    created_by_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    project = relationship("Project", back_populates="jobs")
    image_asset = relationship("ImageAsset", back_populates="jobs")
    created_by = relationship("User", back_populates="jobs")
    artifacts = relationship("Artifact", back_populates="job", cascade="all, delete-orphan")


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    artifact_type = Column(String(100), nullable=False)
    uri = Column(String(1024), nullable=False)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    job = relationship("Job", back_populates="artifacts")
