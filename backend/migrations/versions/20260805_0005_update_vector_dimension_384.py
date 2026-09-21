"""Update vector embedding dimension to 384 for sentence-transformers.

Revision ID: 20260805_0005
Revises: 20260804_0004
Create Date: 2026-08-05
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "20260805_0005"
down_revision: str | None = "20260804_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Alter vector dimension to 384 and rebuild the HNSW index."""
    # Drop existing HNSW index
    op.drop_index("ix_chunks_embedding_hnsw", table_name="document_chunks")

    # Clear existing chunks if any to prevent mixing incompatible vector spaces
    op.execute(sa.text("DELETE FROM document_chunks"))

    # Alter column dimension from 1536 to 384
    op.alter_column(
        "document_chunks",
        "embedding",
        type_=pgvector.sqlalchemy.VECTOR(dim=384),
        existing_type=pgvector.sqlalchemy.VECTOR(dim=1536),
        nullable=False,
    )

    # Recreate HNSW cosine index on the 384-dimensional vector column
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Revert vector dimension back to 1536 and rebuild the HNSW index."""
    op.drop_index("ix_chunks_embedding_hnsw", table_name="document_chunks")
    op.execute(sa.text("DELETE FROM document_chunks"))
    op.alter_column(
        "document_chunks",
        "embedding",
        type_=pgvector.sqlalchemy.VECTOR(dim=1536),
        existing_type=pgvector.sqlalchemy.VECTOR(dim=384),
        nullable=False,
    )
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
