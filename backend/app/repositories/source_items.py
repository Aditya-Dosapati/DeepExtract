"""Persistence operations for external source manifests."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.source_item import SourceItem


async def get_source_item(
    session: AsyncSession,
    owner_id: UUID,
    source_type: str,
    external_id: str,
) -> SourceItem | None:
    """Return one owner-scoped source item by stable external identity."""
    statement = select(SourceItem).where(
        SourceItem.owner_id == owner_id,
        SourceItem.source_type == source_type,
        SourceItem.external_id == external_id,
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def count_document_sources(session: AsyncSession, document_id: UUID) -> int:
    """Count source items still pointing at one canonical document."""
    statement = (
        select(func.count()).select_from(SourceItem).where(SourceItem.document_id == document_id)
    )
    return int((await session.execute(statement)).scalar_one())
