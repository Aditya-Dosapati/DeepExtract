"""Read-only Google Drive MCP adapter for folder synchronization."""

import base64
import binascii
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from mcp import ClientSession

from backend.app.core.config import Settings
from backend.app.mcp.client import authenticated_mcp_session

GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
GOOGLE_DOCUMENT_MIME = "application/vnd.google-apps.document"


class DriveConnectorError(RuntimeError):
    """Safe Google Drive discovery or download failure."""


@dataclass(frozen=True, slots=True)
class DriveFile:
    """Normalized file metadata returned by Drive MCP."""

    id: str
    title: str
    parent_id: str
    mime_type: str
    file_size: int | None
    modified_time: str
    view_url: str | None
    can_add_children: bool

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DriveFile":
        """Validate the minimal metadata needed by the sync pipeline."""
        file_id = str(payload.get("id", "")).strip()
        title = str(payload.get("title", "")).strip()
        mime_type = str(payload.get("mimeType", "")).strip()
        if not file_id or not title or not mime_type:
            raise DriveConnectorError("Drive returned incomplete file metadata")
        raw_size = payload.get("fileSize")
        try:
            file_size = int(str(raw_size)) if raw_size not in {None, ""} else None
        except (TypeError, ValueError) as exc:
            raise DriveConnectorError("Drive returned an invalid file size") from exc
        view_url = payload.get("viewUrl")
        return cls(
            id=file_id,
            title=title,
            parent_id=str(payload.get("parentId", "")),
            mime_type=mime_type,
            file_size=file_size,
            modified_time=str(payload.get("modifiedTime", "")),
            view_url=str(view_url) if view_url else None,
            can_add_children=bool(payload.get("canAddChildren", False)),
        )


@dataclass(frozen=True, slots=True)
class DriveDownload:
    """Downloaded bytes plus the media type used for local parsing."""

    data: bytes
    mime_type: str


class GoogleDriveClient(Protocol):
    """Minimal client contract consumed by the synchronization service."""

    async def list_folder(self, folder_id: str) -> Sequence[DriveFile]: ...

    async def download_file(
        self, file_id: str, export_mime_type: str | None = None
    ) -> DriveDownload: ...


def _structured_content(result: Any) -> dict[str, Any]:
    if bool(getattr(result, "isError", False)):
        raise DriveConnectorError("Google Drive MCP tool call failed")
    structured = getattr(result, "structuredContent", None)
    if not isinstance(structured, Mapping):
        raise DriveConnectorError("Google Drive MCP returned an invalid response")
    return {str(key): value for key, value in structured.items()}


class MCPGoogleDriveClient:
    """Call the official remote Drive MCP server through one initialized session."""

    def __init__(self, session: ClientSession, page_size: int = 100) -> None:
        self.session = session
        self.page_size = page_size

    async def list_folder(self, folder_id: str) -> Sequence[DriveFile]:
        """List every direct child of one folder with MCP pagination."""
        files: list[DriveFile] = []
        page_token: str | None = None
        while True:
            arguments: dict[str, object] = {
                "query": f"parentId = '{folder_id}'",
                "pageSize": self.page_size,
                "excludeContentSnippets": True,
            }
            if page_token:
                arguments["pageToken"] = page_token
            result = await self.session.call_tool("search_files", arguments=arguments)
            payload = _structured_content(result)
            raw_files = payload.get("files", [])
            if not isinstance(raw_files, list):
                raise DriveConnectorError("Google Drive MCP returned an invalid file list")
            for raw_file in raw_files:
                if not isinstance(raw_file, Mapping):
                    raise DriveConnectorError("Google Drive MCP returned invalid file metadata")
                files.append(DriveFile.from_payload(raw_file))
            next_page = payload.get("nextPageToken")
            page_token = str(next_page) if next_page else None
            if page_token is None:
                return files

    async def download_file(
        self, file_id: str, export_mime_type: str | None = None
    ) -> DriveDownload:
        """Download one file as validated base64 bytes."""
        arguments: dict[str, object] = {"fileId": file_id}
        if export_mime_type:
            arguments["exportMimeType"] = export_mime_type
        result = await self.session.call_tool("download_file_content", arguments=arguments)
        payload = _structured_content(result)
        encoded = payload.get("content")
        mime_type = str(payload.get("mimeType") or export_mime_type or "")
        if not isinstance(encoded, str) or not mime_type:
            raise DriveConnectorError("Google Drive MCP returned invalid file content")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise DriveConnectorError("Google Drive MCP returned invalid file content") from exc
        return DriveDownload(data=data, mime_type=mime_type)


@asynccontextmanager
async def open_google_drive_client(
    settings: Settings,
) -> AsyncIterator[GoogleDriveClient]:
    """Open an authenticated Drive MCP client using configured OAuth access."""
    if (
        settings.google_drive_access_token is None
        or not settings.google_drive_access_token.get_secret_value().strip()
    ):
        raise DriveConnectorError("Google Drive access token is not configured")
    token = settings.google_drive_access_token.get_secret_value()
    additional_headers = (
        {"x-goog-user-project": settings.google_drive_quota_project.strip()}
        if settings.google_drive_quota_project.strip()
        else None
    )
    try:
        async with authenticated_mcp_session(
            settings.google_drive_mcp_url,
            token,
            additional_headers=additional_headers,
        ) as session:
            yield MCPGoogleDriveClient(session)
    except DriveConnectorError:
        raise
    except Exception as exc:
        raise DriveConnectorError("Google Drive MCP connection failed") from exc
