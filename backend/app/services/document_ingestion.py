"""Orchestration for validated, deduplicated document ingestion."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import Settings
from backend.app.ingestion.chunking import chunk_pages
from backend.app.ingestion.embeddings import EmbeddingProvider, validate_embeddings
from backend.app.ingestion.extractors import extract_pages, validate_upload
from backend.app.ingestion.normalization import content_hash, normalize_text
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.document_chunk import DocumentChunk
from backend.app.models.ingestion_job import IngestionJob, IngestionJobStatus
from backend.app.repositories.documents import find_document_by_hash


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Document ingestion result including owner-scoped duplicate state."""

    document: Document
    duplicate: bool
    chunks_created: int


class IngestionUnavailableError(ValueError):
    """Raised after an indexing failure has been recorded durably."""


async def ingest_document(
    session: AsyncSession,
    *,
    owner_id: UUID,
    thread_id: UUID | None,
    filename: str,
    mime_type: str,
    data: bytes,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    display_name: str | None = None,
    stored_mime_type: str | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> IngestionResult:
    """Index a document while durably recording running and failed job states."""
    extension = validate_upload(
        filename,
        mime_type,
        data,
        max_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
    )
    pages = extract_pages(extension, data)
    normalized_document = normalize_text("\n\n".join(page.text for page in pages))
    document_hash = content_hash(normalized_document)

    duplicate = await find_document_by_hash(session, owner_id, document_hash)
    if duplicate is not None:
        return IngestionResult(duplicate, duplicate=True, chunks_created=0)

    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("Document did not produce any indexable text chunks")

    safe_filename = Path(filename).name[:255]
    metadata = dict(source_metadata or {})
    metadata.update({"page_count": len(pages), "file_extension": extension})
    document = Document(
        owner_id=owner_id,
        thread_id=thread_id,
        original_filename=safe_filename,
        display_name=(display_name or safe_filename)[:255],
        mime_type=(stored_mime_type or mime_type).lower(),
        file_size=len(data),
        content_hash=document_hash,
        status=DocumentStatus.PROCESSING,
        metadata_json=metadata,
    )
    session.add(document)
    try:
        await session.flush()
        job = IngestionJob(
            document_id=document.id,
            owner_id=owner_id,
            status=IngestionJobStatus.RUNNING,
            details_json={"chunks": len(chunks)},
        )
        session.add(job)
        # This boundary makes the running job observable to concurrent dashboard requests and
        # ensures a later provider failure cannot roll the job record away.
        await session.commit()
    except IntegrityError:
        await session.rollback()
        duplicate = await find_document_by_hash(session, owner_id, document_hash)
        if duplicate is not None:
            return IngestionResult(duplicate, duplicate=True, chunks_created=0)
        raise

    try:
        vectors = await embedding_provider.embed_documents(
            [chunk.normalized_content for chunk in chunks]
        )
        validate_embeddings(vectors, len(chunks), settings.embedding_dimension)
        session.add_all(
            [
                DocumentChunk(
                    document_id=document.id,
                    owner_id=owner_id,
                    thread_id=thread_id,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    section_title=None,
                    content=chunk.content,
                    normalized_content=chunk.normalized_content,
                    content_hash=chunk.content_hash,
                    token_count=chunk.token_count,
                    embedding=vector,
                    metadata_json={**(source_metadata or {}), **chunk.metadata},
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
        )
        document.status = DocumentStatus.COMPLETED
        job.status = IngestionJobStatus.COMPLETED
        await session.commit()
    except Exception as exc:
        await session.rollback()
        failed_document = await session.get(Document, document.id)
        failed_job = await session.get(IngestionJob, job.id)
        if failed_document is not None:
            failed_document.status = DocumentStatus.FAILED
        if failed_job is not None:
            failed_job.status = IngestionJobStatus.FAILED
            failed_job.error_message = "Document indexing failed"
        await session.commit()
        raise IngestionUnavailableError("Document indexing failed") from exc

    await session.refresh(document)
    return IngestionResult(document, duplicate=False, chunks_created=len(chunks))
