"""Semantic retrieval request and result schemas."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """Owner-scoped semantic search parameters."""

    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    thread_id: UUID | None = None
    metadata: dict[str, Any] | None = None


class RetrievalResult(BaseModel):
    """One source-addressable vector search result."""

    document_id: UUID
    document_name: str
    page_number: int | None
    chunk_id: UUID
    chunk_index: int
    score: float = Field(ge=0, le=1)
    excerpt: str


class RetrievalResponse(BaseModel):
    """Ranked source chunks for a query."""

    results: list[RetrievalResult]
