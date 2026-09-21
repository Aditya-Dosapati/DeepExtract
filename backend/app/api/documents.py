"""Authenticated document upload and metadata endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, Response, UploadFile, status

from backend.app.auth.dependencies import CurrentUserDep, SettingsDep
from backend.app.core.errors import ApplicationError
from backend.app.db.session import SessionDep
from backend.app.ingestion.embeddings import create_embedding_provider
from backend.app.ingestion.extractors import DocumentValidationError
from backend.app.repositories import documents as document_repository
from backend.app.repositories.threads import get_thread
from backend.app.schemas.document import DocumentPublic, DocumentUploadResponse
from backend.app.services.document_ingestion import IngestionUnavailableError, ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File()],
    user: CurrentUserDep,
    settings: SettingsDep,
    session: SessionDep,
    thread_id: Annotated[UUID | None, Form()] = None,
) -> DocumentUploadResponse:
    """Validate and synchronously index one owner-scoped in-memory upload."""
    if thread_id is not None and await get_thread(session, user.id, thread_id) is None:
        raise ApplicationError(404, "thread_not_found", "Conversation not found")
    data = await file.read(settings.max_upload_size_mb * 1024 * 1024 + 1)
    try:
        provider = create_embedding_provider(settings)
        result = await ingest_document(
            session,
            owner_id=user.id,
            thread_id=thread_id,
            filename=file.filename or "upload",
            mime_type=file.content_type or "application/octet-stream",
            data=data,
            settings=settings,
            embedding_provider=provider,
        )
    except DocumentValidationError as exc:
        raise ApplicationError(422, "invalid_document", str(exc)) from exc
    except IngestionUnavailableError as exc:
        raise ApplicationError(503, "ingestion_unavailable", str(exc)) from exc
    return DocumentUploadResponse(
        document=DocumentPublic.model_validate(result.document),
        duplicate=result.duplicate,
        chunks_created=result.chunks_created,
    )


@router.get("", response_model=list[DocumentPublic])
async def list_documents(user: CurrentUserDep, session: SessionDep) -> list[DocumentPublic]:
    """List only the authenticated user's documents."""
    documents = await document_repository.list_documents(session, user.id)
    return [DocumentPublic.model_validate(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentPublic)
async def get_document(
    document_id: UUID, user: CurrentUserDep, session: SessionDep
) -> DocumentPublic:
    """Return one owned document without disclosing foreign IDs."""
    document = await document_repository.get_document(session, user.id, document_id)
    if document is None:
        raise ApplicationError(404, "document_not_found", "Document not found")
    return DocumentPublic.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: UUID, user: CurrentUserDep, session: SessionDep) -> Response:
    """Delete one owned document and cascaded vector data."""
    deleted = await document_repository.delete_document(session, user.id, document_id)
    if not deleted:
        await session.rollback()
        raise ApplicationError(404, "document_not_found", "Document not found")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
