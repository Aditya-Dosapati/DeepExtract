"""Dashboard metric response schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class KnowledgeBaseMetrics(BaseModel):
    """PostgreSQL-backed knowledge-base inventory for one owner."""

    documents: int = Field(ge=0)
    indexed_documents: int = Field(ge=0)
    processing_documents: int = Field(ge=0)
    failed_documents: int = Field(ge=0)
    indexed_chunks: int = Field(ge=0)
    stored_bytes: int = Field(ge=0)
    last_indexed_at: datetime | None


class ActivityMetrics(BaseModel):
    """Conversation and ingestion activity for one owner."""

    threads: int = Field(ge=0)
    messages: int = Field(ge=0)
    active_ingestion_jobs: int = Field(ge=0)
    failed_ingestion_jobs: int = Field(ge=0)


class MetricsOverview(BaseModel):
    """Complete owner-scoped dashboard snapshot."""

    knowledge_base: KnowledgeBaseMetrics
    activity: ActivityMetrics
