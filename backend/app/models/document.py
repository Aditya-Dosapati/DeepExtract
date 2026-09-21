"""Uploaded document metadata model."""

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, BigInteger, Enum, ForeignKey, Index, String, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base, TimestampMixin


class DocumentStatus(StrEnum):
    """Document ingestion lifecycle states."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(TimestampMixin, Base):
    """Owner-scoped metadata for one uploaded document."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("owner_id", "content_hash", name="uq_documents_owner_content_hash"),
        Index("ix_documents_owner_created", "owner_id", "created_at"),
        Index("ix_documents_thread_created", "thread_id", "created_at"),
        Index("ix_documents_metadata_gin", "metadata_json", postgresql_using="gin"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    thread_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversation_threads.id", ondelete="CASCADE"),
        nullable=True,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            name="document_status",
            native_enum=False,
            values_callable=lambda enum: [member.value for member in enum],
        ),
        default=DocumentStatus.PENDING,
        nullable=False,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict, nullable=False
    )
