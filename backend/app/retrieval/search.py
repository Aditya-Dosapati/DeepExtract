"""Database-native owner-scoped pgvector similarity search."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import cast, or_, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_chunk import DocumentChunk


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Internal typed result returned by the vector query."""

    chunk: DocumentChunk
    document_name: str
    score: float


async def search_chunks(
    session: AsyncSession,
    *,
    owner_id: UUID,
    query_vector: list[float],
    top_k: int,
    score_threshold: float,
    thread_id: UUID | None = None,
    include_global: bool = False,
    metadata: dict[str, Any] | None = None,
) -> list[SearchHit]:
    """Rank chunks in PostgreSQL after applying tenant and optional scope filters."""
    if session.bind is not None and session.bind.dialect.name == "sqlite":
        statement = (
            select(DocumentChunk, Document.display_name)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                DocumentChunk.owner_id == owner_id,
                Document.owner_id == owner_id,
                Document.status == DocumentStatus.COMPLETED,
            )
        )
        if thread_id is not None:
            thread_filter = (
                or_(DocumentChunk.thread_id == thread_id, DocumentChunk.thread_id.is_(None))
                if include_global
                else DocumentChunk.thread_id == thread_id
            )
            statement = statement.where(thread_filter)
        if metadata:
            statement = statement.where(DocumentChunk.metadata_json.contains(metadata))

        rows_sqlite = (await session.execute(statement)).all()
        scored: list[tuple[DocumentChunk, str, float]] = []
        for chunk_item, doc_name in rows_sqlite:
            vec = chunk_item.embedding
            if isinstance(vec, str):
                import json

                vec = json.loads(vec)
            dot = sum(a * b for a, b in zip(vec, query_vector, strict=True))
            sim = max(0.0, min(1.0, float(dot)))
            if sim >= score_threshold:
                scored.append((chunk_item, str(doc_name), sim))
        scored.sort(key=lambda item: item[2], reverse=True)
        return [
            SearchHit(chunk_val, doc_name_val, sim_val)
            for chunk_val, doc_name_val, sim_val in scored[:top_k]
        ]

    distance = DocumentChunk.embedding.cosine_distance(query_vector)
    score = (1 - distance).label("similarity_score")
    statement = (
        select(DocumentChunk, Document.display_name, score)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.owner_id == owner_id,
            Document.owner_id == owner_id,
            Document.status == DocumentStatus.COMPLETED,
            distance <= 1 - score_threshold,
        )
        .order_by(distance)
        .limit(top_k)
    )
    if thread_id is not None:
        thread_filter = (
            or_(DocumentChunk.thread_id == thread_id, DocumentChunk.thread_id.is_(None))
            if include_global
            else DocumentChunk.thread_id == thread_id
        )
        statement = statement.where(thread_filter)
    if metadata:
        metadata_column = (
            cast(DocumentChunk.metadata_json, JSONB)
            if session.bind is not None and session.bind.dialect.name == "postgresql"
            else DocumentChunk.metadata_json
        )
        statement = statement.where(metadata_column.contains(metadata))

    rows = (await session.execute(statement)).all()
    seen: set[UUID] = set()
    hits: list[SearchHit] = []
    for chunk, document_name, similarity_score in rows:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        hits.append(
            SearchHit(chunk, str(document_name), max(0.0, min(1.0, float(similarity_score))))
        )
    return hits
