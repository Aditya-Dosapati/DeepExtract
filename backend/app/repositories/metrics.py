"""Owner-scoped operational and knowledge-base aggregates."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_chunk import DocumentChunk
from backend.app.models.ingestion_job import IngestionJob, IngestionJobStatus
from backend.app.models.message import Message
from backend.app.models.thread import ConversationThread


@dataclass(frozen=True, slots=True)
class OwnerMetrics:
    """Single-query dashboard values for one authenticated owner."""

    documents: int
    indexed_documents: int
    processing_documents: int
    failed_documents: int
    indexed_chunks: int
    stored_bytes: int
    threads: int
    messages: int
    active_ingestion_jobs: int
    failed_ingestion_jobs: int
    last_indexed_at: datetime | None


async def get_owner_metrics(session: AsyncSession, owner_id: UUID) -> OwnerMetrics:
    """Compute dashboard metrics without exposing another owner's activity."""
    document_statement = select(
        func.count(Document.id),
        func.coalesce(func.sum(case((Document.status == DocumentStatus.COMPLETED, 1), else_=0)), 0),
        func.coalesce(
            func.sum(case((Document.status == DocumentStatus.PROCESSING, 1), else_=0)), 0
        ),
        func.coalesce(func.sum(case((Document.status == DocumentStatus.FAILED, 1), else_=0)), 0),
        func.coalesce(func.sum(Document.file_size), 0),
        func.max(case((Document.status == DocumentStatus.COMPLETED, Document.updated_at))),
    ).where(Document.owner_id == owner_id)
    documents = (await session.execute(document_statement)).one()

    chunks = await session.scalar(
        select(func.count(DocumentChunk.id)).where(DocumentChunk.owner_id == owner_id)
    )
    threads = await session.scalar(
        select(func.count(ConversationThread.id)).where(ConversationThread.owner_id == owner_id)
    )
    messages = await session.scalar(
        select(func.count(Message.id)).where(Message.owner_id == owner_id)
    )
    jobs = (
        await session.execute(
            select(
                func.coalesce(
                    func.sum(
                        case(
                            (
                                IngestionJob.status.in_(
                                    [IngestionJobStatus.PENDING, IngestionJobStatus.RUNNING]
                                ),
                                1,
                            ),
                            else_=0,
                        )
                    ),
                    0,
                ),
                func.coalesce(
                    func.sum(case((IngestionJob.status == IngestionJobStatus.FAILED, 1), else_=0)),
                    0,
                ),
            ).where(IngestionJob.owner_id == owner_id)
        )
    ).one()
    return OwnerMetrics(
        documents=int(documents[0]),
        indexed_documents=int(documents[1]),
        processing_documents=int(documents[2]),
        failed_documents=int(documents[3]),
        stored_bytes=int(documents[4]),
        last_indexed_at=documents[5],
        indexed_chunks=int(chunks or 0),
        threads=int(threads or 0),
        messages=int(messages or 0),
        active_ingestion_jobs=int(jobs[0]),
        failed_ingestion_jobs=int(jobs[1]),
    )
