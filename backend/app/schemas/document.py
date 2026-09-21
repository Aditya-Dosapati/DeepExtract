"""Document ingestion and metadata response schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.app.models.document import DocumentStatus


class DocumentPublic(BaseModel):
    """Safe owner-visible document metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    thread_id: UUID | None
    original_filename: str
    display_name: str
    mime_type: str
    file_size: int
    content_hash: str
    status: DocumentStatus
    metadata_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DocumentUploadResponse(BaseModel):
    """Result of an upload and synchronous ingestion attempt."""

    document: DocumentPublic
    duplicate: bool
    chunks_created: int = Field(ge=0)
