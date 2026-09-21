"""Authenticated semantic retrieval endpoint."""

from fastapi import APIRouter

from backend.app.auth.dependencies import CurrentUserDep, SettingsDep
from backend.app.core.errors import ApplicationError
from backend.app.db.session import SessionDep
from backend.app.ingestion.embeddings import create_embedding_provider, validate_embeddings
from backend.app.repositories.threads import get_thread
from backend.app.retrieval.search import search_chunks
from backend.app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievalResult

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalResponse)
async def semantic_search(
    payload: RetrievalRequest,
    user: CurrentUserDep,
    settings: SettingsDep,
    session: SessionDep,
) -> RetrievalResponse:
    """Embed a query and search only vectors owned by the authenticated user."""
    query = payload.query.strip()
    if not query:
        raise ApplicationError(422, "invalid_query", "Query cannot be blank")
    if (
        payload.thread_id is not None
        and await get_thread(session, user.id, payload.thread_id) is None
    ):
        raise ApplicationError(404, "thread_not_found", "Conversation not found")
    try:
        provider = create_embedding_provider(settings)
        vectors = await provider.embed_documents([query])
        validate_embeddings(vectors, 1, settings.embedding_dimension)
    except ValueError as exc:
        raise ApplicationError(503, "retrieval_unavailable", str(exc)) from exc
    hits = await search_chunks(
        session,
        owner_id=user.id,
        query_vector=vectors[0],
        top_k=payload.top_k or settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
        thread_id=payload.thread_id,
        include_global=payload.thread_id is not None,
        metadata=payload.metadata,
    )
    results: list[RetrievalResult] = []
    context_characters = 0
    for hit in hits:
        excerpt = hit.chunk.content[:1000]
        if results and context_characters + len(excerpt) > settings.retrieval_max_context_chars:
            break
        context_characters += len(excerpt)
        results.append(
            RetrievalResult(
                document_id=hit.chunk.document_id,
                document_name=hit.document_name,
                page_number=hit.chunk.page_number,
                chunk_id=hit.chunk.id,
                chunk_index=hit.chunk.chunk_index,
                score=hit.score,
                excerpt=excerpt,
            )
        )
    return RetrievalResponse(results=results)
