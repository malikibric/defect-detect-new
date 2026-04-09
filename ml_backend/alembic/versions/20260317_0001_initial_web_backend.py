"""Initial web backend schema

Revision ID: 20260317_0001
Revises: 
Create Date: 2026-03-17
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260317_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("is_superuser", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(op.f("ix_users_id"), "users", ["id"], unique=False)
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "projects",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index(op.f("ix_projects_id"), "projects", ["id"], unique=False)
    op.create_index(op.f("ix_projects_owner_id"), "projects", ["owner_id"], unique=False)

    op.create_table(
        "image_assets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=True),
        sa.Column("source_uri", sa.String(length=1024), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=True),
        sa.Column("checksum", sa.String(length=128), nullable=True),
        sa.Column("original_filename", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )
    op.create_index(op.f("ix_image_assets_id"), "image_assets", ["id"], unique=False)
    op.create_index(op.f("ix_image_assets_owner_id"), "image_assets", ["owner_id"], unique=False)
    op.create_index(op.f("ix_image_assets_project_id"), "image_assets", ["project_id"], unique=False)

    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("type", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column("webhook_url", sa.String(length=1024), nullable=True),
        sa.Column("project_id", sa.Integer(), nullable=True),
        sa.Column("image_asset_id", sa.Integer(), nullable=True),
        sa.Column("created_by_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["image_asset_id"], ["image_assets.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="SET NULL"),
    )
    op.create_index(op.f("ix_jobs_id"), "jobs", ["id"], unique=False)
    op.create_index(op.f("ix_jobs_type"), "jobs", ["type"], unique=False)
    op.create_index(op.f("ix_jobs_status"), "jobs", ["status"], unique=False)
    op.create_index(op.f("ix_jobs_project_id"), "jobs", ["project_id"], unique=False)
    op.create_index(op.f("ix_jobs_image_asset_id"), "jobs", ["image_asset_id"], unique=False)
    op.create_index(op.f("ix_jobs_created_by_id"), "jobs", ["created_by_id"], unique=False)
    op.create_index(op.f("ix_jobs_celery_task_id"), "jobs", ["celery_task_id"], unique=False)

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.Integer(), nullable=False),
        sa.Column("artifact_type", sa.String(length=100), nullable=False),
        sa.Column("uri", sa.String(length=1024), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
    )
    op.create_index(op.f("ix_artifacts_id"), "artifacts", ["id"], unique=False)
    op.create_index(op.f("ix_artifacts_job_id"), "artifacts", ["job_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_artifacts_job_id"), table_name="artifacts")
    op.drop_index(op.f("ix_artifacts_id"), table_name="artifacts")
    op.drop_table("artifacts")

    op.drop_index(op.f("ix_jobs_celery_task_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_created_by_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_image_asset_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_project_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_status"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_type"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_id"), table_name="jobs")
    op.drop_table("jobs")

    op.drop_index(op.f("ix_image_assets_owner_id"), table_name="image_assets")
    op.drop_index(op.f("ix_image_assets_project_id"), table_name="image_assets")
    op.drop_index(op.f("ix_image_assets_id"), table_name="image_assets")
    op.drop_table("image_assets")

    op.drop_index(op.f("ix_projects_owner_id"), table_name="projects")
    op.drop_index(op.f("ix_projects_id"), table_name="projects")
    op.drop_table("projects")

    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_index(op.f("ix_users_id"), table_name="users")
    op.drop_table("users")
