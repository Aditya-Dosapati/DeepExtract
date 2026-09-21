"""Focused Google Drive folder synchronization tests."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.connectors.google_drive import (
    GOOGLE_DOCUMENT_MIME,
    GOOGLE_FOLDER_MIME,
    DriveConnectorError,
    DriveDownload,
    DriveFile,
)
from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.ingestion.embeddings import DeterministicEmbeddingProvider
from backend.app.models.document import Document
from backend.app.models.document_chunk import DocumentChunk
from backend.app.models.source_item import SourceItem, SourceItemStatus
from backend.app.models.user import User
from backend.app.services.google_drive_sync import (
    discover_folder,
    route_drive_file,
    sync_google_drive_folder,
)


def drive_file(
    file_id: str,
    title: str,
    mime_type: str = "text/plain",
    *,
    parent_id: str = "root",
    version: str = "v1",
    folder: bool = False,
) -> DriveFile:
    """Build deterministic Drive metadata for tests."""
    return DriveFile(
        id=file_id,
        title=title,
        parent_id=parent_id,
        mime_type=mime_type,
        file_size=None if folder else 100,
        modified_time=version,
        view_url=f"https://drive.test/{file_id}",
        can_add_children=folder,
    )


class FakeDriveClient:
    """In-memory Drive tree and content transport."""

    def __init__(
        self,
        folders: dict[str, list[DriveFile]],
        downloads: dict[str, bytes | Exception],
    ) -> None:
        self.folders = folders
        self.downloads = downloads
        self.listed: list[str] = []
        self.downloaded: list[tuple[str, str | None]] = []

    async def list_folder(self, folder_id: str) -> Sequence[DriveFile]:
        self.listed.append(folder_id)
        return self.folders.get(folder_id, [])

    async def download_file(
        self, file_id: str, export_mime_type: str | None = None
    ) -> DriveDownload:
        self.downloaded.append((file_id, export_mime_type))
        value = self.downloads[file_id]
        if isinstance(value, Exception):
            raise value
        return DriveDownload(value, export_mime_type or "text/plain")


@asynccontextmanager
async def database_session() -> AsyncIterator[AsyncSession]:
    """Create one isolated relational store for a sync scenario."""
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        yield session
    await engine.dispose()


def drive_settings() -> Settings:
    """Return settings for the fixed test folder."""
    return Settings(
        app_env="test",
        database_url_override=SecretStr("sqlite+aiosqlite://"),
        jwt_secret=SecretStr("unit-test-secret-with-at-least-32-characters"),
        embedding_provider="fake",
        llm_provider="fake",
        google_drive_folder_id="root",
        google_drive_access_token=SecretStr("test-token"),
    )


async def create_owner(session: AsyncSession) -> User:
    owner = User(username="owner", email="owner@example.com", password_hash="hash")
    session.add(owner)
    await session.commit()
    return owner


def test_format_routing_is_intentionally_narrow() -> None:
    google_doc = drive_file("doc", "Policy", GOOGLE_DOCUMENT_MIME)
    pdf = drive_file("pdf", "Policy", "application/pdf")
    sheet = drive_file("sheet", "Budget", "application/vnd.google-apps.spreadsheet")

    google_route = route_drive_file(google_doc)
    pdf_route = route_drive_file(pdf)

    assert google_route is not None
    assert google_route.filename == "Policy.txt"
    assert google_route.export_mime_type == "text/plain"
    assert pdf_route is not None
    assert pdf_route.filename == "Policy.pdf"
    assert route_drive_file(sheet) is None


async def test_discovery_stays_inside_root_and_recurses() -> None:
    child = drive_file("child", "Policies", GOOGLE_FOLDER_MIME, folder=True)
    root_file = drive_file("root-file", "root.txt")
    nested_file = drive_file("nested-file", "nested.txt", parent_id="child")
    client = FakeDriveClient(
        {"root": [child, root_file], "child": [nested_file], "outside": []},
        {},
    )

    files = await discover_folder(client, "root", include_subfolders=True, max_files=10)

    assert [file.id for file in files] == ["root-file", "nested-file"]
    assert client.listed == ["root", "child"]


async def test_sync_deduplicates_content_and_skips_unchanged_versions() -> None:
    first = drive_file("first", "policy-a.txt")
    second = drive_file("second", "policy-b.txt")
    unsupported = drive_file("sheet", "budget", "application/vnd.google-apps.spreadsheet")
    client = FakeDriveClient(
        {"root": [first, second, unsupported]},
        {
            "first": b"Employees receive twenty leave days.",
            "second": b"Employees receive twenty leave days.",
        },
    )
    settings = drive_settings()
    provider = DeterministicEmbeddingProvider(settings.embedding_dimension)
    async with database_session() as session:
        owner = await create_owner(session)

        initial = await sync_google_drive_folder(session, client, settings, provider, owner.id)
        repeated = await sync_google_drive_folder(session, client, settings, provider, owner.id)
        document_count = await session.scalar(select(func.count()).select_from(Document))
        chunk_count = await session.scalar(select(func.count()).select_from(DocumentChunk))
        items = (await session.execute(select(SourceItem))).scalars().all()

    assert initial.discovered == 3
    assert initial.indexed == 2
    assert initial.duplicates == 1
    assert initial.skipped == 1
    assert repeated.unchanged == 2
    assert client.downloaded == [("first", None), ("second", None)]
    assert document_count == 1
    assert chunk_count == 1
    assert {item.status for item in items} == {
        SourceItemStatus.INDEXED,
        SourceItemStatus.SKIPPED,
    }


async def test_changed_file_replaces_orphan_and_failures_are_isolated() -> None:
    changing = drive_file("changing", "policy.txt")
    broken = drive_file("broken", "broken.txt")
    client = FakeDriveClient(
        {"root": [changing, broken]},
        {
            "changing": b"Initial travel policy.",
            "broken": DriveConnectorError("Drive download failed"),
        },
    )
    settings = drive_settings()
    provider = DeterministicEmbeddingProvider(settings.embedding_dimension)
    async with database_session() as session:
        owner = await create_owner(session)
        first = await sync_google_drive_folder(session, client, settings, provider, owner.id)
        client.folders["root"][0] = drive_file("changing", "policy.txt", version="v2")
        client.downloads["changing"] = b"Updated travel policy with new limits."

        second = await sync_google_drive_folder(session, client, settings, provider, owner.id)
        documents = (await session.execute(select(Document))).scalars().all()
        failed = await session.scalar(select(SourceItem).where(SourceItem.external_id == "broken"))

    assert first.indexed == 1
    assert first.failed == 1
    assert second.indexed == 1
    assert second.failed == 1
    assert len(documents) == 1
    assert documents[0].display_name == "policy.txt"
    assert failed is not None
    assert failed.status == SourceItemStatus.FAILED
    assert failed.error_message == "Drive download failed"
