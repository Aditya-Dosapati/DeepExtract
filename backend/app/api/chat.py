"""Authenticated agent conversation endpoints."""

import json
import logging
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from backend.app.agents.dependencies import CheckpointerDep
from backend.app.auth.dependencies import CurrentUserDep, SettingsDep
from backend.app.core.errors import ApplicationError
from backend.app.db.session import DatabaseDep, SessionDep
from backend.app.repositories.messages import list_messages
from backend.app.repositories.threads import get_thread
from backend.app.schemas.chat import ChatRequest, ChatResponse, MessagePublic, SourceCitation
from backend.app.services.chat import answer_question, stream_answer_question

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/{thread_id}", response_model=ChatResponse)
async def chat(
    thread_id: UUID,
    payload: ChatRequest,
    user: CurrentUserDep,
    settings: SettingsDep,
    checkpointer: CheckpointerDep,
    session: SessionDep,
) -> ChatResponse:
    """Answer from documents scoped to one owned conversation."""
    if await get_thread(session, user.id, thread_id) is None:
        raise ApplicationError(404, "thread_not_found", "Conversation not found")
    question = payload.question.strip()
    if not question:
        raise ApplicationError(422, "invalid_question", "Question cannot be blank")
    try:
        message, state = await answer_question(
            session,
            owner_id=user.id,
            thread_id=thread_id,
            question=question,
            settings=settings,
            checkpointer=checkpointer,
        )
    except ValueError as exc:
        raise ApplicationError(503, "agent_unavailable", str(exc)) from exc
    await session.commit()
    return ChatResponse(
        thread_id=thread_id,
        message_id=message.id,
        answer=message.content,
        grounded=state["grounded"],
        sources=[SourceCitation.model_validate(source) for source in state["sources"]],
    )


@router.post("/{thread_id}/stream", response_class=StreamingResponse)
async def stream_chat(
    thread_id: UUID,
    payload: ChatRequest,
    user: CurrentUserDep,
    settings: SettingsDep,
    checkpointer: CheckpointerDep,
    database: DatabaseDep,
    session: SessionDep,
) -> StreamingResponse:
    """Stream one grounded answer as server-sent events and persist its final state."""
    if await get_thread(session, user.id, thread_id) is None:
        raise ApplicationError(404, "thread_not_found", "Conversation not found")
    question = payload.question.strip()
    if not question:
        raise ApplicationError(422, "invalid_question", "Question cannot be blank")

    async def events() -> AsyncIterator[str]:
        try:
            async with database.sessions() as stream_session:
                async for event in stream_answer_question(
                    stream_session,
                    owner_id=user.id,
                    thread_id=thread_id,
                    question=question,
                    settings=settings,
                    checkpointer=checkpointer,
                ):
                    payload_json = json.dumps(jsonable_encoder(event["data"]))
                    yield f"event: {event['event']}\ndata: {payload_json}\n\n"
        except Exception as exc:
            logger.exception("Agent stream failed for thread %s", thread_id, exc_info=exc)
            error_json = json.dumps({"message": "The agent could not complete this response."})
            yield f"event: error\ndata: {error_json}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{thread_id}/history", response_model=list[MessagePublic])
async def chat_history(
    thread_id: UUID, user: CurrentUserDep, session: SessionDep
) -> list[MessagePublic]:
    """Return chronological history for one owned conversation."""
    if await get_thread(session, user.id, thread_id) is None:
        raise ApplicationError(404, "thread_not_found", "Conversation not found")
    messages = await list_messages(session, user.id, thread_id)
    return [MessagePublic.model_validate(message) for message in messages]
