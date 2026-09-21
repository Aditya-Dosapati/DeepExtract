"""Owner-scoped MCP document tool implementations."""

import base64
import binascii
from typing import Any
from uuid import UUID

from mcp.server.fastmcp.exceptions import ToolError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agents.checkpoints import AgentCheckpointer
from backend.app.core.config import Settings
from backend.app.ingestion.embeddings import create_embedding_provider, validate_embeddings
from backend.app.ingestion.extractors import DocumentValidationError
from backend.app.models.user import User
from backend.app.repositories import documents as document_repository
from backend.app.repositories.threads import get_thread
from backend.app.repositories.users import get_user_by_id
from backend.app.retrieval.search import search_chunks
from backend.app.schemas.document import DocumentPublic
from backend.app.services.chat import answer_question
from backend.app.services.document_ingestion import IngestionUnavailableError, ingest_document


async def require_active_user(session: AsyncSession, subject: str) -> User:
    """Resolve and re-check the active user for every tool invocation."""
    try:
        user_id = UUID(subject)
    except ValueError as exc:
        raise ToolError("Authentication subject is invalid") from exc
    user = await get_user_by_id(session, user_id)
    if user is None or not user.is_active:
        raise ToolError("Authentication credentials are invalid or expired")
    return user


async def list_documents_tool(session: AsyncSession, subject: str) -> dict[str, Any]:
    """List document metadata for the authenticated MCP principal."""
    user = await require_active_user(session, subject)
    documents = await document_repository.list_documents(session, user.id)
    return {
        "documents": [
            DocumentPublic.model_validate(document).model_dump(mode="json")
            for document in documents
        ]
    }


async def get_document_tool(
    session: AsyncSession, subject: str, document_id: UUID
) -> dict[str, Any]:
    """Return an owned document or a non-disclosing not-found error."""
    user = await require_active_user(session, subject)
    document = await document_repository.get_document(session, user.id, document_id)
    if document is None:
        raise ToolError("Document not found")
    return {"document": DocumentPublic.model_validate(document).model_dump(mode="json")}


async def search_documents_tool(
    session: AsyncSession,
    settings: Settings,
    subject: str,
    query: str,
    top_k: int | None,
    thread_id: UUID | None,
) -> dict[str, Any]:
    """Run database-native semantic search for the authenticated principal."""
    user = await require_active_user(session, subject)
    normalized_query = query.strip()
    if not normalized_query:
        raise ToolError("Query cannot be blank")
    if thread_id is not None and await get_thread(session, user.id, thread_id) is None:
        raise ToolError("Conversation not found")
    provider = create_embedding_provider(settings)
    vectors = await provider.embed_documents([normalized_query])
    validate_embeddings(vectors, 1, settings.embedding_dimension)
    hits = await search_chunks(
        session,
        owner_id=user.id,
        query_vector=vectors[0],
        top_k=min(top_k or settings.retrieval_top_k, 100),
        score_threshold=settings.retrieval_score_threshold,
        thread_id=thread_id,
    )
    return {
        "results": [
            {
                "document_id": str(hit.chunk.document_id),
                "document_name": hit.document_name,
                "page_number": hit.chunk.page_number,
                "chunk_id": str(hit.chunk.id),
                "chunk_index": hit.chunk.chunk_index,
                "score": hit.score,
                "excerpt": hit.chunk.content[:1000],
            }
            for hit in hits
        ]
    }


async def answer_from_documents_tool(
    session: AsyncSession,
    settings: Settings,
    subject: str,
    query: str,
    thread_id: UUID,
    checkpointer: AgentCheckpointer | None = None,
) -> dict[str, Any]:
    """Run the persisted, owner-scoped answering graph for an MCP principal."""
    user = await require_active_user(session, subject)
    if await get_thread(session, user.id, thread_id) is None:
        raise ToolError("Conversation not found")
    normalized_query = query.strip()
    if not normalized_query:
        raise ToolError("Query cannot be blank")
    try:
        message, state = await answer_question(
            session,
            owner_id=user.id,
            thread_id=thread_id,
            question=normalized_query,
            settings=settings,
            checkpointer=checkpointer,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    await session.commit()
    return {
        "thread_id": str(thread_id),
        "message_id": str(message.id),
        "answer": message.content,
        "grounded": state["grounded"],
        "sources": state["sources"],
    }


async def ingest_document_tool(
    session: AsyncSession,
    settings: Settings,
    subject: str,
    *,
    filename: str,
    mime_type: str,
    content_base64: str,
    thread_id: UUID | None,
) -> dict[str, Any]:
    """Decode an in-request upload and ingest it without accepting filesystem paths."""
    user = await require_active_user(session, subject)
    if thread_id is not None and await get_thread(session, user.id, thread_id) is None:
        raise ToolError("Conversation not found")
    try:
        data = base64.b64decode(content_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ToolError("Document content must be valid base64") from exc
    try:
        result = await ingest_document(
            session,
            owner_id=user.id,
            thread_id=thread_id,
            filename=filename,
            mime_type=mime_type,
            data=data,
            settings=settings,
            embedding_provider=create_embedding_provider(settings),
        )
    except (DocumentValidationError, IngestionUnavailableError) as exc:
        raise ToolError(str(exc)) from exc
    return {
        "document": DocumentPublic.model_validate(result.document).model_dump(mode="json"),
        "duplicate": result.duplicate,
        "chunks_created": result.chunks_created,
    }
