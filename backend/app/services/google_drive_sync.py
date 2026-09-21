"""Google Drive folder discovery and durable document synchronization."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.connectors.google_drive import (
    GOOGLE_DOCUMENT_MIME,
    GOOGLE_FOLDER_MIME,
    DriveConnectorError,
    DriveFile,
    GoogleDriveClient,
)
from backend.app.core.config import Settings
from backend.app.ingestion.embeddings import EmbeddingProvider
from backend.app.ingestion.extractors import DocumentValidationError
from backend.app.models.source_item import SourceItem, SourceItemStatus
from backend.app.repositories import documents as document_repository
from backend.app.repositories import source_items as source_item_repository
from backend.app.services.document_ingestion import IngestionUnavailableError, ingest_document

SOURCE_TYPE = "google_drive"

SUPPORTED_BINARY_TYPES: dict[str, tuple[str, str]] = {
    "application/pdf": (".pdf", "application/pdf"),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    "text/plain": (".txt", "text/plain"),
}


@dataclass(frozen=True, slots=True)
class DriveParseRoute:
    """How one Drive MIME type enters the existing upload parser."""

    filename: str
    parser_mime_type: str
    export_mime_type: str | None = None


@dataclass(slots=True)
class DriveSyncSummary:
    """Observable counts for one manual folder synchronization."""

    folder_id: str
    discovered: int = 0
    indexed: int = 0
    unchanged: int = 0
    duplicates: int = 0
    skipped: int = 0
    failed: int = 0


def route_drive_file(file: DriveFile) -> DriveParseRoute | None:
    """Return a deterministic parser route for supported first-slice formats."""
    if file.mime_type == GOOGLE_DOCUMENT_MIME:
        title = file.title if file.title.lower().endswith(".txt") else f"{file.title}.txt"
        return DriveParseRoute(title, "text/plain", "text/plain")
    route = SUPPORTED_BINARY_TYPES.get(file.mime_type)
    if route is None:
        return None
    extension, parser_mime_type = route
    filename = file.title
    if Path(filename).suffix.lower() != extension:
        filename = f"{filename}{extension}"
    return DriveParseRoute(filename, parser_mime_type)


async def discover_folder(
    client: GoogleDriveClient,
    folder_id: str,
    *,
    include_subfolders: bool,
    max_files: int,
) -> list[DriveFile]:
    """Traverse only descendants of the configured root folder."""
    pending = [folder_id]
    visited_folders: set[str] = set()
    discovered: list[DriveFile] = []
    seen_files: set[str] = set()
    while pending:
        current_folder = pending.pop(0)
        if current_folder in visited_folders:
            continue
        visited_folders.add(current_folder)
        for item in await client.list_folder(current_folder):
            if item.mime_type == GOOGLE_FOLDER_MIME or item.can_add_children:
                if include_subfolders:
                    pending.append(item.id)
                continue
            if item.id in seen_files:
                continue
            seen_files.add(item.id)
            discovered.append(item)
            if len(discovered) > max_files:
                raise DriveConnectorError("Google Drive folder exceeds the configured file limit")
    return discovered


async def _get_or_create_source_item(
    session: AsyncSession,
    owner_id: UUID,
    folder_id: str,
    file: DriveFile,
) -> SourceItem:
    item = await source_item_repository.get_source_item(session, owner_id, SOURCE_TYPE, file.id)
    if item is None:
        item = SourceItem(
            owner_id=owner_id,
            source_type=SOURCE_TYPE,
            folder_id=folder_id,
            external_id=file.id,
            external_version=file.modified_time,
            file_name=file.title[:255],
            mime_type=file.mime_type,
            view_url=file.view_url,
            status=SourceItemStatus.DISCOVERED,
            metadata_json={"parent_id": file.parent_id},
        )
        session.add(item)
    else:
        item.folder_id = folder_id
        item.file_name = file.title[:255]
        item.mime_type = file.mime_type
        item.view_url = file.view_url
        item.metadata_json = {"parent_id": file.parent_id}
    await session.commit()
    return item


async def _mark_source_failure(
    session: AsyncSession, item_id: UUID, version: str, message: str
) -> None:
    await session.rollback()
    item = await session.get(SourceItem, item_id)
    if item is None:
        return
    item.external_version = version
    item.status = SourceItemStatus.FAILED
    item.error_message = message
    await session.commit()


async def _remove_orphaned_drive_document(
    session: AsyncSession, owner_id: UUID, document_id: UUID
) -> None:
    if await source_item_repository.count_document_sources(session, document_id) != 0:
        return
    document = await document_repository.get_document(session, owner_id, document_id)
    if document is None or document.metadata_json.get("source_type") != SOURCE_TYPE:
        return
    await document_repository.delete_document(session, owner_id, document_id)
    await session.commit()


async def _sync_file(
    session: AsyncSession,
    client: GoogleDriveClient,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    owner_id: UUID,
    folder_id: str,
    file: DriveFile,
    summary: DriveSyncSummary,
) -> None:
    item = await _get_or_create_source_item(session, owner_id, folder_id, file)
    if (
        item.status == SourceItemStatus.INDEXED
        and item.document_id is not None
        and item.external_version == file.modified_time
    ):
        summary.unchanged += 1
        return

    route = route_drive_file(file)
    if route is None:
        item.external_version = file.modified_time
        item.status = SourceItemStatus.SKIPPED
        item.error_message = "Unsupported Google Drive file type"
        await session.commit()
        summary.skipped += 1
        return
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file.file_size is not None and file.file_size > max_size:
        item.external_version = file.modified_time
        item.status = SourceItemStatus.SKIPPED
        item.error_message = "Google Drive file exceeds the configured size limit"
        await session.commit()
        summary.skipped += 1
        return

    old_document_id = item.document_id
    try:
        download = await client.download_file(file.id, route.export_mime_type)
        source_metadata = {
            "source_type": SOURCE_TYPE,
            "source_file_id": file.id,
            "source_folder_id": folder_id,
            "source_url": file.view_url,
            "source_modified_time": file.modified_time,
            "source_mime_type": file.mime_type,
        }
        result = await ingest_document(
            session,
            owner_id=owner_id,
            thread_id=None,
            filename=route.filename,
            mime_type=route.parser_mime_type,
            data=download.data,
            settings=settings,
            embedding_provider=embedding_provider,
            display_name=file.title,
            stored_mime_type=file.mime_type,
            source_metadata=source_metadata,
        )
    except (
        DriveConnectorError,
        DocumentValidationError,
        IngestionUnavailableError,
        ValueError,
    ) as exc:
        await _mark_source_failure(session, item.id, file.modified_time, str(exc))
        summary.failed += 1
        return
    except Exception:
        await _mark_source_failure(
            session, item.id, file.modified_time, "Google Drive file synchronization failed"
        )
        summary.failed += 1
        return

    refreshed_item = await session.get(SourceItem, item.id)
    if refreshed_item is None:
        raise RuntimeError("Source item disappeared during synchronization")
    refreshed_item.document_id = result.document.id
    refreshed_item.external_version = file.modified_time
    refreshed_item.content_hash = result.document.content_hash
    refreshed_item.status = SourceItemStatus.INDEXED
    refreshed_item.error_message = None
    await session.commit()
    summary.indexed += 1
    if result.duplicate:
        summary.duplicates += 1
    if old_document_id is not None and old_document_id != result.document.id:
        await _remove_orphaned_drive_document(session, owner_id, old_document_id)


async def sync_google_drive_folder(
    session: AsyncSession,
    client: GoogleDriveClient,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    owner_id: UUID,
) -> DriveSyncSummary:
    """Synchronize the configured folder while isolating individual file failures."""
    folder_id = settings.google_drive_folder_id.strip()
    if not folder_id:
        raise DriveConnectorError("Google Drive folder ID is not configured")
    files = await discover_folder(
        client,
        folder_id,
        include_subfolders=settings.google_drive_include_subfolders,
        max_files=settings.google_drive_max_files,
    )
    summary = DriveSyncSummary(folder_id=folder_id, discovered=len(files))
    for file in files:
        await _sync_file(
            session,
            client,
            settings,
            embedding_provider,
            owner_id,
            folder_id,
            file,
            summary,
        )
    return summary
