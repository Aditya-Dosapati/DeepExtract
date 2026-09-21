"""Google Drive MCP response parsing and pagination tests."""

import base64
from typing import Any, cast

import pytest
from mcp import ClientSession

from backend.app.connectors.google_drive import (
    DriveConnectorError,
    MCPGoogleDriveClient,
)


class FakeResult:
    """Minimal MCP call result with structured output."""

    def __init__(self, payload: dict[str, Any], *, is_error: bool = False) -> None:
        self.structuredContent = payload
        self.isError = is_error


class FakeSession:
    """Scripted MCP session used to observe tool arguments."""

    def __init__(self, responses: list[FakeResult]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, name: str, arguments: dict[str, object]) -> FakeResult:
        self.calls.append((name, arguments))
        return self.responses.pop(0)


async def test_drive_client_paginates_folder_children() -> None:
    session = FakeSession(
        [
            FakeResult(
                {
                    "files": [
                        {
                            "id": "one",
                            "title": "one.txt",
                            "parentId": "folder",
                            "mimeType": "text/plain",
                            "fileSize": "3",
                            "modifiedTime": "v1",
                        }
                    ],
                    "nextPageToken": "next",
                }
            ),
            FakeResult({"files": []}),
        ]
    )
    client = MCPGoogleDriveClient(cast(ClientSession, session), page_size=25)

    files = await client.list_folder("folder")

    assert [file.id for file in files] == ["one"]
    assert session.calls == [
        (
            "search_files",
            {
                "query": "parentId = 'folder'",
                "pageSize": 25,
                "excludeContentSnippets": True,
            },
        ),
        (
            "search_files",
            {
                "query": "parentId = 'folder'",
                "pageSize": 25,
                "excludeContentSnippets": True,
                "pageToken": "next",
            },
        ),
    ]


async def test_drive_client_decodes_download_content() -> None:
    session = FakeSession(
        [
            FakeResult(
                {
                    "id": "doc",
                    "title": "Policy",
                    "mimeType": "text/plain",
                    "content": base64.b64encode(b"Business policy").decode(),
                }
            )
        ]
    )
    client = MCPGoogleDriveClient(cast(ClientSession, session))

    download = await client.download_file("doc", "text/plain")

    assert download.data == b"Business policy"
    assert download.mime_type == "text/plain"
    assert session.calls == [
        (
            "download_file_content",
            {"fileId": "doc", "exportMimeType": "text/plain"},
        )
    ]


async def test_drive_client_rejects_tool_errors() -> None:
    session = FakeSession([FakeResult({}, is_error=True)])
    client = MCPGoogleDriveClient(cast(ClientSession, session))

    with pytest.raises(DriveConnectorError, match="tool call failed"):
        await client.list_folder("folder")
