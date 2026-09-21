"""External source items synchronized into canonical documents."""

from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Enum, ForeignKey, Index, String, Text, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base, TimestampMixin


class SourceItemStatus(StrEnum):
    """Lifecycle state for one discovered external file."""

    DISCOVERED = "discovered"
    INDEXED = "indexed"
    SKIPPED = "skipped"
    FAILED = "failed"


class SourceItem(TimestampMixin, Base):
    """Stable mapping from an external file to an indexed document."""

    __tablename__ = "source_items"
    __table_args__ = (
        UniqueConstraint(
            "owner_id",
            "source_type",
            "external_id",
            name="uq_source_items_owner_source_external",
        ),
        Index("ix_source_items_owner_folder", "owner_id", "folder_id"),
        Index("ix_source_items_document", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    document_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    folder_id: Mapped[str] = mapped_column(String(255), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    external_version: Mapped[str] = mapped_column(String(100), default="", nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(150), nullable=False)
    view_url: Mapped[str | None] = mapped_column(Text(), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[SourceItemStatus] = mapped_column(
        Enum(
            SourceItemStatus,
            name="source_item_status",
            native_enum=False,
            values_callable=lambda enum: [member.value for member in enum],
        ),
        default=SourceItemStatus.DISCOVERED,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(Text(), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict, nullable=False
    )
