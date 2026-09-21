"""Owner-scoped document persistence operations."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.document import Document
from backend.app.models.document_chunk import DocumentChunk
from backend.app.models.ingestion_job import IngestionJob


async def find_document_by_hash(
    session: AsyncSession, owner_id: UUID, document_hash: str
) -> Document | None:
    """Find an owner's existing normalized document without crossing tenants."""
    statement = select(Document).where(
        Document.owner_id == owner_id,
        Document.content_hash == document_hash,
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def list_documents(session: AsyncSession, owner_id: UUID) -> Sequence[Document]:
    """List documents belonging to one authenticated owner."""
    statement = (
        select(Document).where(Document.owner_id == owner_id).order_by(Document.created_at.desc())
    )
    return (await session.execute(statement)).scalars().all()


async def get_document(session: AsyncSession, owner_id: UUID, document_id: UUID) -> Document | None:
    """Return an owned document or nothing for both missing and foreign IDs."""
    statement = select(Document).where(
        Document.id == document_id,
        Document.owner_id == owner_id,
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def delete_document(session: AsyncSession, owner_id: UUID, document_id: UUID) -> bool:
    """Delete one owned document and its cascaded chunks and jobs."""
    await session.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.owner_id == owner_id,
        )
    )
    await session.execute(
        delete(IngestionJob).where(
            IngestionJob.document_id == document_id,
            IngestionJob.owner_id == owner_id,
        )
    )
    statement = delete(Document).where(
        Document.id == document_id,
        Document.owner_id == owner_id,
    )
    result = await session.execute(statement.returning(Document.id))
    return result.scalar_one_or_none() is not None
