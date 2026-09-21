"""Authenticated MCP client context tests."""

from typing import Any

from backend.app.mcp import client as mcp_client


async def test_authenticated_client_initializes_and_yields_session(monkeypatch: Any) -> None:
    observed: dict[str, Any] = {}

    class FakeHTTPClient:
        def __init__(self, **kwargs: Any) -> None:
            observed["headers"] = kwargs["headers"]

        async def __aenter__(self) -> "FakeHTTPClient":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class FakeTransport:
        async def __aenter__(self) -> tuple[str, str, None]:
            return "read", "write", None

        async def __aexit__(self, *_args: Any) -> None:
            return None

    class FakeSession:
        initialized = False

        def __init__(self, read: str, write: str) -> None:
            observed["streams"] = (read, write)

        async def __aenter__(self) -> "FakeSession":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def initialize(self) -> None:
            self.initialized = True
            observed["initialized"] = True

    monkeypatch.setattr("backend.app.mcp.client.httpx.AsyncClient", FakeHTTPClient)
    monkeypatch.setattr(
        mcp_client,
        "streamable_http_client",
        lambda _url, **_kwargs: FakeTransport(),
    )
    monkeypatch.setattr(mcp_client, "ClientSession", FakeSession)

    async with mcp_client.authenticated_mcp_session("https://mcp.test", "secret-token") as session:
        assert session is not None

    assert observed == {
        "headers": {"Authorization": "Bearer secret-token"},
        "streams": ("read", "write"),
        "initialized": True,
    }
