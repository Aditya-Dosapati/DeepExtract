"""List tools and documents through the authenticated MCP client."""

import asyncio
import os

from backend.app.mcp.client import authenticated_mcp_session


async def run() -> None:
    """Connect with environment-provided credentials and print non-secret results."""
    token = os.environ.get("MCP_ACCESS_TOKEN")
    if not token:
        raise SystemExit("MCP_ACCESS_TOKEN is required")
    url = os.environ.get("MCP_SERVER_URL", "http://localhost:8001/mcp")
    async with authenticated_mcp_session(url, token) as session:
        listed = await session.list_tools()
        documents = await session.call_tool("list_documents", arguments={})
        query = os.environ.get("MCP_SMOKE_QUERY")
        search = (
            await session.call_tool("search_documents", arguments={"query": query})
            if query
            else None
        )
    print("tools:", ", ".join(tool.name for tool in listed.tools))
    print("documents:", documents.structuredContent)
    if search is not None:
        print("search:", search.structuredContent)


if __name__ == "__main__":
    asyncio.run(run())
