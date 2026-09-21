"""Owner-scoped conversation thread endpoints."""

from uuid import UUID

from fastapi import APIRouter, Response, status

from backend.app.agents.checkpoints import checkpoint_thread_id
from backend.app.agents.dependencies import CheckpointerDep
from backend.app.auth.dependencies import CurrentUserDep
from backend.app.core.errors import ApplicationError
from backend.app.db.session import SessionDep
from backend.app.repositories import threads as thread_repository
from backend.app.schemas.thread import ThreadCreate, ThreadPublic

router = APIRouter(prefix="/threads", tags=["threads"])


@router.post("", response_model=ThreadPublic, status_code=status.HTTP_201_CREATED)
async def create_thread(
    payload: ThreadCreate, user: CurrentUserDep, session: SessionDep
) -> ThreadPublic:
    """Create a conversation owned by the authenticated user."""
    thread = await thread_repository.create_thread(session, user.id, payload.title.strip())
    await session.commit()
    return ThreadPublic.model_validate(thread)


@router.get("", response_model=list[ThreadPublic])
async def list_threads(user: CurrentUserDep, session: SessionDep) -> list[ThreadPublic]:
    """List the authenticated user's conversations."""
    threads = await thread_repository.list_threads(session, user.id)
    return [ThreadPublic.model_validate(thread) for thread in threads]


@router.get("/{thread_id}", response_model=ThreadPublic)
async def get_thread(thread_id: UUID, user: CurrentUserDep, session: SessionDep) -> ThreadPublic:
    """Return one owned conversation without exposing foreign records."""
    thread = await thread_repository.get_thread(session, user.id, thread_id)
    if thread is None:
        raise ApplicationError(404, "thread_not_found", "Conversation not found")
    return ThreadPublic.model_validate(thread)


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(
    thread_id: UUID,
    user: CurrentUserDep,
    checkpointer: CheckpointerDep,
    session: SessionDep,
) -> Response:
    """Delete one owned conversation and its dependent resources."""
    deleted = await thread_repository.delete_thread(session, user.id, thread_id)
    if not deleted:
        await session.rollback()
        raise ApplicationError(404, "thread_not_found", "Conversation not found")
    await checkpointer.adelete_thread(checkpoint_thread_id(user.id, thread_id))
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
