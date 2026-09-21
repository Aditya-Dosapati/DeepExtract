"""Transactional conversation orchestration."""

from collections.abc import AsyncIterator
from typing import cast
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agents.checkpoints import AgentCheckpointer
from backend.app.agents.providers import create_answer_provider
from backend.app.agents.types import AgentStreamEvent
from backend.app.agents.workflow import AgentState, run_agent, stream_agent
from backend.app.core.config import Settings
from backend.app.ingestion.embeddings import create_embedding_provider
from backend.app.models.message import Message, MessageRole
from backend.app.repositories.messages import create_message, list_messages


async def answer_question(
    session: AsyncSession,
    *,
    owner_id: UUID,
    thread_id: UUID,
    question: str,
    settings: Settings,
    checkpointer: AgentCheckpointer | None = None,
) -> tuple[Message, AgentState]:
    """Persist a user turn, run the graph, and persist its validated response."""
    history = list(await list_messages(session, owner_id, thread_id))
    await create_message(
        session,
        thread_id=thread_id,
        owner_id=owner_id,
        role=MessageRole.USER,
        content=question,
    )
    state = await run_agent(
        session=session,
        owner_id=owner_id,
        thread_id=thread_id,
        query=question,
        settings=settings,
        embedding_provider=create_embedding_provider(settings),
        answer_provider=create_answer_provider(settings),
        history=history,
        checkpointer=checkpointer,
    )
    assistant = await create_message(
        session,
        thread_id=thread_id,
        owner_id=owner_id,
        role=MessageRole.ASSISTANT,
        content=state["answer"],
        sources=state["sources"],
    )
    return assistant, state


async def stream_answer_question(
    session: AsyncSession,
    *,
    owner_id: UUID,
    thread_id: UUID,
    question: str,
    settings: Settings,
    checkpointer: AgentCheckpointer | None = None,
) -> AsyncIterator[AgentStreamEvent]:
    """Persist a user turn, stream graph tokens, then persist the validated answer."""
    history = list(await list_messages(session, owner_id, thread_id))
    await create_message(
        session,
        thread_id=thread_id,
        owner_id=owner_id,
        role=MessageRole.USER,
        content=question,
    )
    await session.commit()
    async for event in stream_agent(
        session=session,
        owner_id=owner_id,
        thread_id=thread_id,
        query=question,
        settings=settings,
        embedding_provider=create_embedding_provider(settings),
        answer_provider=create_answer_provider(settings),
        history=history,
        checkpointer=checkpointer,
    ):
        if event["event"] == "token":
            yield event
            continue
        data = event["data"]
        if not isinstance(data, dict):
            raise RuntimeError("Agent completed without a final state")
        state = cast(AgentState, data)
        assistant = await create_message(
            session,
            thread_id=thread_id,
            owner_id=owner_id,
            role=MessageRole.ASSISTANT,
            content=state["answer"],
            sources=state["sources"],
        )
        await session.commit()
        yield {
            "event": "complete",
            "data": {
                "thread_id": str(thread_id),
                "message_id": str(assistant.id),
                "answer": assistant.content,
                "grounded": state["grounded"],
                "sources": state["sources"],
            },
        }
