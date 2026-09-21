"""Factory and entrypoint for the authenticated Streamable HTTP MCP server."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import UUID

import anyio
import uvicorn
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from backend.app.agents.checkpoints import AgentCheckpointer, create_checkpointer
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import configure_logging
from backend.app.db.session import Database
from backend.app.mcp.auth import ApplicationTokenVerifier, authenticated_subject
from backend.app.mcp.tools import (
    answer_from_documents_tool,
    get_document_tool,
    ingest_document_tool,
    list_documents_tool,
    search_documents_tool,
)


@dataclass(slots=True)
class MCPApplicationContext:
    """Infrastructure owned by one running MCP server."""

    settings: Settings
    database: Database
    checkpointer: AgentCheckpointer


MCPContext = Context[ServerSession, MCPApplicationContext]


def create_mcp_server(settings_override: Settings | None = None) -> FastMCP[MCPApplicationContext]:
    """Create the MCP server without starting a transport during import."""
    settings = settings_override or get_settings()

    @asynccontextmanager
    async def lifespan(
        _server: FastMCP[MCPApplicationContext],
    ) -> AsyncIterator[MCPApplicationContext]:
        database = Database(settings.database_url)
        try:
            async with create_checkpointer(settings) as checkpointer:
                yield MCPApplicationContext(
                    settings=settings,
                    database=database,
                    checkpointer=checkpointer,
                )
        finally:
            with anyio.CancelScope(shield=True):
                await database.close()

    server = FastMCP[MCPApplicationContext](
        name="Agentic RAG Knowledge Assistant",
        instructions=(
            "Tools operate only on the authenticated user's knowledge base. Use "
            "search_documents for source-addressable evidence and answer_from_documents for a "
            "complete grounded answer with validated sources. Use ingest_document only when the "
            "user explicitly asks to index supplied file content. Never infer identifiers or "
            "retry non-transient tool errors."
        ),
        token_verifier=ApplicationTokenVerifier(settings),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(settings.mcp_issuer_url),
            resource_server_url=AnyHttpUrl(settings.mcp_resource_server_url),
            required_scopes=["user"],
        ),
        host=settings.mcp_host,
        port=settings.mcp_port,
        streamable_http_path=settings.mcp_path,
        stateless_http=True,
        json_response=True,
        lifespan=lifespan,
    )

    @server.tool(name="list_documents", structured_output=True)
    async def list_documents(ctx: MCPContext) -> dict[str, object]:
        """List authenticated-user document metadata; returns {documents: [...]} without text."""
        application = ctx.request_context.lifespan_context
        async with application.database.sessions() as session:
            return await list_documents_tool(session, authenticated_subject())

    @server.tool(name="get_document", structured_output=True)
    async def get_document(document_id: UUID, ctx: MCPContext) -> dict[str, object]:
        """Get owned document metadata by UUID; returns {document: {...}} or not found."""
        application = ctx.request_context.lifespan_context
        async with application.database.sessions() as session:
            return await get_document_tool(session, authenticated_subject(), document_id)

    @server.tool(name="search_documents", structured_output=True)
    async def search_documents(
        query: str,
        ctx: MCPContext,
        top_k: int | None = None,
        thread_id: UUID | None = None,
    ) -> dict[str, object]:
        """Semantic search returning ranked chunk excerpts, scores, and source identifiers."""
        application = ctx.request_context.lifespan_context
        async with application.database.sessions() as session:
            return await search_documents_tool(
                session,
                application.settings,
                authenticated_subject(),
                query,
                top_k,
                thread_id,
            )

    @server.tool(name="ingest_document", structured_output=True)
    async def ingest_document(
        filename: str,
        mime_type: str,
        content_base64: str,
        ctx: MCPContext,
        thread_id: UUID | None = None,
    ) -> dict[str, object]:
        """Index explicit base64 file content and return document and chunk counts."""
        application = ctx.request_context.lifespan_context
        async with application.database.sessions() as session:
            return await ingest_document_tool(
                session,
                application.settings,
                authenticated_subject(),
                filename=filename,
                mime_type=mime_type,
                content_base64=content_base64,
                thread_id=thread_id,
            )

    @server.tool(name="answer_from_documents", structured_output=True)
    async def answer_from_documents(
        query: str,
        thread_id: UUID,
        ctx: MCPContext,
    ) -> dict[str, object]:
        """Run grounded RAG for an owned thread; returns answer, grounding status, and sources."""
        application = ctx.request_context.lifespan_context
        async with application.database.sessions() as session:
            return await answer_from_documents_tool(
                session,
                application.settings,
                authenticated_subject(),
                query,
                thread_id,
                application.checkpointer,
            )

    return server


def main() -> None:
    """Start the configured MCP transport only when explicitly invoked."""
    settings = get_settings()
    configure_logging(settings.log_level)
    server = create_mcp_server(settings)
    if settings.mcp_transport == "stdio":
        server.run(transport="stdio")
        return

    async def health(_request: Request) -> JSONResponse:
        database = Database(settings.database_url)
        try:
            ready = await database.is_ready()
        finally:
            await database.close()
        return JSONResponse(
            {"status": "ready" if ready else "unavailable", "service": "mcp"},
            status_code=200 if ready else 503,
        )

    app = server.streamable_http_app()
    app.routes.insert(0, Route("/health", health, methods=["GET"]))
    uvicorn.run(
        app,
        host=settings.mcp_host,
        port=settings.mcp_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
