"""Add external source item tracking.

Revision ID: 20260804_0004
Revises: 20260804_0003
Create Date: 2026-08-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260804_0004"
down_revision: str | None = "20260804_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the external source manifest table."""
    op.create_table(
        "source_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("folder_id", sa.String(length=255), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("external_version", sa.String(length=100), nullable=False),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=150), nullable=False),
        sa.Column("view_url", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "discovered",
                "indexed",
                "skipped",
                "failed",
                name="source_item_status",
                native_enum=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_source_items_document_id_documents"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name=op.f("fk_source_items_owner_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_source_items")),
        sa.UniqueConstraint(
            "owner_id",
            "source_type",
            "external_id",
            name="uq_source_items_owner_source_external",
        ),
    )
    op.create_index("ix_source_items_owner_folder", "source_items", ["owner_id", "folder_id"])
    op.create_index("ix_source_items_document", "source_items", ["document_id"])


def downgrade() -> None:
    """Remove external source tracking."""
    op.drop_index("ix_source_items_document", table_name="source_items")
    op.drop_index("ix_source_items_owner_folder", table_name="source_items")
    op.drop_table("source_items")
