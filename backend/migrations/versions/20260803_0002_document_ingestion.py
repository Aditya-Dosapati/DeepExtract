"""Create document ingestion and vector retrieval schema.

Revision ID: 20260803_0002
Revises: 20260803_0001
Create Date: 2026-08-03
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260803_0002"
down_revision: str | None = "20260803_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create owner-scoped document, chunk, embedding, and job tables."""
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=True),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name=op.f("fk_documents_owner_id_users"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["conversation_threads.id"],
            name=op.f("fk_documents_thread_id_conversation_threads"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("owner_id", "content_hash", name="uq_documents_owner_content_hash"),
    )
    op.create_index("ix_documents_owner_created", "documents", ["owner_id", "created_at"])
    op.create_index("ix_documents_thread_created", "documents", ["thread_id", "created_at"])
    op.create_index(
        "ix_documents_metadata_gin", "documents", ["metadata_json"], postgresql_using="gin"
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("thread_id", sa.Uuid(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("section_title", sa.String(length=500), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("normalized_content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.VECTOR(dim=1536), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name=op.f("fk_document_chunks_owner_id_users"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["conversation_threads.id"],
            name=op.f("fk_document_chunks_thread_id_conversation_threads"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_chunks")),
        sa.UniqueConstraint("document_id", "content_hash", name="uq_chunks_document_content_hash"),
    )
    op.create_index(
        "ix_chunks_document_index", "document_chunks", ["document_id", "chunk_index"], unique=True
    )
    op.create_index("ix_chunks_owner_thread", "document_chunks", ["owner_id", "thread_id"])
    op.create_index(
        "ix_chunks_metadata_gin", "document_chunks", ["metadata_json"], postgresql_using="gin"
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("details_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_ingestion_jobs_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name=op.f("fk_ingestion_jobs_owner_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_jobs")),
    )
    op.create_index("ix_ingestion_jobs_owner_created", "ingestion_jobs", ["owner_id", "created_at"])


def downgrade() -> None:
    """Remove document ingestion and retrieval schema."""
    op.drop_index("ix_ingestion_jobs_owner_created", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
    op.drop_index("ix_chunks_embedding_hnsw", table_name="document_chunks")
    op.drop_index("ix_chunks_metadata_gin", table_name="document_chunks")
    op.drop_index("ix_chunks_owner_thread", table_name="document_chunks")
    op.drop_index("ix_chunks_document_index", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_index("ix_documents_metadata_gin", table_name="documents")
    op.drop_index("ix_documents_thread_created", table_name="documents")
    op.drop_index("ix_documents_owner_created", table_name="documents")
    op.drop_table("documents")
