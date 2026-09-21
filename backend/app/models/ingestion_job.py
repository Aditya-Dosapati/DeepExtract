"""Document ingestion job status model."""

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Enum, ForeignKey, Index, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base, TimestampMixin


class IngestionJobStatus(StrEnum):
    """Ingestion job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJob(TimestampMixin, Base):
    """Observable ingestion attempt for one authorized document."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (Index("ix_ingestion_jobs_owner_created", "owner_id", "created_at"),)

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[IngestionJobStatus] = mapped_column(
        Enum(
            IngestionJobStatus,
            name="ingestion_job_status",
            native_enum=False,
            values_callable=lambda enum: [member.value for member in enum],
        ),
        default=IngestionJobStatus.PENDING,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict, nullable=False
    )
