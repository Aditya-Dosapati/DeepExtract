"""Agent chat request, history, and citation schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.app.models.message import MessageRole


class ChatRequest(BaseModel):
    """One question submitted to an existing conversation."""

    question: str = Field(min_length=1, max_length=4000)


class SourceCitation(BaseModel):
    """A source record constructed from an authorized database chunk."""

    document_id: UUID
    document_name: str
    page_number: int | None
    chunk_id: UUID
    chunk_index: int
    score: float = Field(ge=0, le=1)
    excerpt: str


class MessagePublic(BaseModel):
    """Owner-visible persisted conversation message."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    thread_id: UUID
    role: MessageRole
    content: str
    sources: list[dict[str, Any]]
    created_at: datetime


class ChatResponse(BaseModel):
    """Validated answer with database-owned citations."""

    thread_id: UUID
    message_id: UUID
    answer: str
    grounded: bool
    sources: list[SourceCitation]
