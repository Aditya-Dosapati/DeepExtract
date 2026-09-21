"""Small authenticated client for the application's Streamable HTTP MCP server."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


@asynccontextmanager
async def authenticated_mcp_session(
    url: str,
    access_token: str,
    *,
    additional_headers: Mapping[str, str] | None = None,
) -> AsyncIterator[ClientSession]:
    """Connect, authenticate, and initialize an MCP client session."""
    headers = {"Authorization": f"Bearer {access_token}"}
    if additional_headers:
        headers.update(additional_headers)
    async with httpx.AsyncClient(
        headers=headers,
        timeout=30,
    ) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
