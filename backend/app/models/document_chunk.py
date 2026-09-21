"""Normalized document chunk and embedding model."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Text, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base, utc_now

EMBEDDING_DIMENSION = 384


class DocumentChunk(Base):
    """One retrievable, source-addressable document segment."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "content_hash", name="uq_chunks_document_content_hash"),
        Index("ix_chunks_owner_thread", "owner_id", "thread_id"),
        Index("ix_chunks_document_index", "document_id", "chunk_index", unique=True),
        Index("ix_chunks_metadata_gin", "metadata_json", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    owner_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    thread_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversation_threads.id", ondelete="CASCADE"),
        nullable=True,
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    page_number: Mapped[int | None] = mapped_column(nullable=True)
    section_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        VECTOR(EMBEDDING_DIMENSION).with_variant(JSON, "sqlite"), nullable=False
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
