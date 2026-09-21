"""Authenticated MCP Streamable HTTP integration tests."""

import base64
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest_asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import SecretStr
from starlette.applications import Starlette

from backend.app.auth.security import create_access_token
from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.db.session import Database
from backend.app.mcp.server import create_mcp_server
from backend.app.models.thread import ConversationThread
from backend.app.models.user import User


@dataclass(frozen=True, slots=True)
class MCPTestEnvironment:
    app: Starlette
    settings: Settings
    alice: User
    bob: User
    alice_thread: ConversationThread


@pytest_asyncio.fixture
async def mcp_environment(tmp_path: Path) -> MCPTestEnvironment:
    """Create an MCP ASGI server with one active user in a file-backed SQLite database."""
    database_path = str(tmp_path) + "/mcp-test.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    settings = Settings(
        app_env="test",
        database_url_override=SecretStr(database_url),
        jwt_secret=SecretStr("mcp-test-secret-with-at-least-32-characters"),
        embedding_provider="fake",
        llm_provider="fake",
    )
    setup_database = Database(database_url)
    async with setup_database.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with setup_database.sessions() as session:
        alice = User(
            username="mcp-user",
            email="mcp-user@example.com",
            password_hash="not-used-by-mcp",
        )
        bob = User(
            username="mcp-other-user",
            email="mcp-other-user@example.com",
            password_hash="not-used-by-mcp",
        )
        session.add_all([alice, bob])
        await session.flush()
        alice_thread = ConversationThread(owner_id=alice.id, title="MCP questions")
        session.add(alice_thread)
        await session.commit()
        await session.refresh(alice)
        await session.refresh(bob)
        await session.refresh(alice_thread)
    await setup_database.close()

    server = create_mcp_server(settings)
    app = server.streamable_http_app()
    return MCPTestEnvironment(
        app=app,
        settings=settings,
        alice=alice,
        bob=bob,
        alice_thread=alice_thread,
    )


async def test_mcp_requires_bearer_authentication(mcp_environment: MCPTestEnvironment) -> None:
    transport = httpx.ASGITransport(app=mcp_environment.app)
    async with mcp_environment.app.router.lifespan_context(mcp_environment.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/mcp", json={})

    assert response.status_code == 401
    assert "Bearer" in response.headers["www-authenticate"]


async def test_mcp_lists_and_calls_only_safe_owner_scoped_tools(
    mcp_environment: MCPTestEnvironment,
) -> None:
    token, _ = create_access_token(mcp_environment.alice.id, mcp_environment.settings)
    transport = httpx.ASGITransport(app=mcp_environment.app)
    async with mcp_environment.app.router.lifespan_context(mcp_environment.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        ) as http_client:
            async with streamable_http_client(
                "http://test/mcp", http_client=http_client, terminate_on_close=False
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    result = await session.call_tool("list_documents", arguments={})

    tools = {tool.name: tool for tool in listed.tools}
    assert set(tools) == {
        "answer_from_documents",
        "get_document",
        "ingest_document",
        "list_documents",
        "search_documents",
    }
    for tool in tools.values():
        properties = tool.inputSchema.get("properties", {})
        assert "user_id" not in properties
        assert "access_token" not in properties
        assert "path" not in properties
    assert result.isError is False
    assert result.structuredContent == {"documents": []}


async def test_mcp_ingestion_and_document_access_remain_owner_scoped(
    mcp_environment: MCPTestEnvironment,
) -> None:
    alice_token, _ = create_access_token(mcp_environment.alice.id, mcp_environment.settings)
    bob_token, _ = create_access_token(mcp_environment.bob.id, mcp_environment.settings)
    transport = httpx.ASGITransport(app=mcp_environment.app)
    encoded = base64.b64encode(b"MCP private policy content").decode()

    async with mcp_environment.app.router.lifespan_context(mcp_environment.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": f"Bearer {alice_token}"},
        ) as alice_http:
            async with streamable_http_client(
                "http://test/mcp", http_client=alice_http, terminate_on_close=False
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    ingested = await session.call_tool(
                        "ingest_document",
                        arguments={
                            "filename": "private.txt",
                            "mime_type": "text/plain",
                            "content_base64": encoded,
                        },
                    )
                    listed = await session.call_tool("list_documents", arguments={})

        assert ingested.isError is False
        assert ingested.structuredContent is not None
        document = ingested.structuredContent["document"]
        assert isinstance(document, dict)
        document_id = document["id"]
        assert listed.structuredContent is not None
        assert len(listed.structuredContent["documents"]) == 1

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": f"Bearer {bob_token}"},
        ) as bob_http:
            async with streamable_http_client(
                "http://test/mcp", http_client=bob_http, terminate_on_close=False
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    foreign = await session.call_tool(
                        "get_document", arguments={"document_id": document_id}
                    )

    assert foreign.isError is True


async def test_mcp_answer_tool_runs_the_persisted_agent_directly(
    mcp_environment: MCPTestEnvironment,
) -> None:
    token, _ = create_access_token(mcp_environment.alice.id, mcp_environment.settings)
    transport = httpx.ASGITransport(app=mcp_environment.app)

    async with mcp_environment.app.router.lifespan_context(mcp_environment.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        ) as http_client:
            async with streamable_http_client(
                "http://test/mcp", http_client=http_client, terminate_on_close=False
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "answer_from_documents",
                        arguments={
                            "query": "Hello",
                            "thread_id": str(mcp_environment.alice_thread.id),
                        },
                    )

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["grounded"] is False
    assert result.structuredContent["sources"] == []
    assert str(result.structuredContent["answer"]).startswith("Hello.")
