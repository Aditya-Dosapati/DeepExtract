"""Owner-scoped conversation message persistence."""

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.message import Message, MessageRole


async def create_message(
    session: AsyncSession,
    *,
    thread_id: UUID,
    owner_id: UUID,
    role: MessageRole,
    content: str,
    sources: list[dict[str, Any]] | None = None,
) -> Message:
    """Persist one message under an already authorized owner and thread."""
    message = Message(
        thread_id=thread_id,
        owner_id=owner_id,
        role=role,
        content=content,
        sources=sources or [],
    )
    session.add(message)
    await session.flush()
    await session.refresh(message)
    return message


async def list_messages(
    session: AsyncSession, owner_id: UUID, thread_id: UUID
) -> Sequence[Message]:
    """Return chronological history after applying both ownership predicates."""
    statement = (
        select(Message)
        .where(Message.owner_id == owner_id, Message.thread_id == thread_id)
        .order_by(Message.created_at, Message.id)
    )
    return (await session.execute(statement)).scalars().all()
