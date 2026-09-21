"""Owner-scoped conversation thread persistence."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.thread import ConversationThread


async def create_thread(session: AsyncSession, owner_id: UUID, title: str) -> ConversationThread:
    """Create a thread owned by the authenticated user."""
    thread = ConversationThread(owner_id=owner_id, title=title)
    session.add(thread)
    await session.flush()
    await session.refresh(thread)
    return thread


async def list_threads(session: AsyncSession, owner_id: UUID) -> Sequence[ConversationThread]:
    """List only threads owned by the authenticated user."""
    statement = (
        select(ConversationThread)
        .where(ConversationThread.owner_id == owner_id)
        .order_by(ConversationThread.created_at.desc())
    )
    return (await session.execute(statement)).scalars().all()


async def get_thread(
    session: AsyncSession, owner_id: UUID, thread_id: UUID
) -> ConversationThread | None:
    """Fetch a thread with ownership applied in the database query."""
    statement = select(ConversationThread).where(
        ConversationThread.id == thread_id,
        ConversationThread.owner_id == owner_id,
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def delete_thread(session: AsyncSession, owner_id: UUID, thread_id: UUID) -> bool:
    """Delete an owned thread without disclosing foreign records."""
    statement = delete(ConversationThread).where(
        ConversationThread.id == thread_id,
        ConversationThread.owner_id == owner_id,
    )
    result = await session.execute(statement.returning(ConversationThread.id))
    return result.scalar_one_or_none() is not None
