"""FastAPI application factory and ASGI entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.agents.checkpoints import create_checkpointer
from backend.app.api import (
    auth,
    chat,
    data_sources,
    documents,
    health,
    metrics,
    retrieval,
    threads,
    users,
)
from backend.app.core.config import Settings, get_settings
from backend.app.core.errors import register_exception_handlers
from backend.app.core.logging import configure_logging
from backend.app.db.session import Database

API_PREFIX = "/api/v1"


def create_app(settings_override: Settings | None = None) -> FastAPI:
    """Create an isolated application, loading infrastructure only in its lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = settings_override or get_settings()
        configure_logging(settings.log_level)
        app.state.settings = settings
        app.state.database = Database(settings.database_url)
        # Pre-warm local embedding provider to prevent cold-start latency
        if settings.embedding_provider in {"sentence_transformers", "huggingface"}:
            from backend.app.ingestion.embeddings import create_embedding_provider

            create_embedding_provider(settings)
        try:
            async with create_checkpointer(settings) as checkpointer:
                app.state.checkpointer = checkpointer
                yield
        finally:
            await app.state.database.close()

    application = FastAPI(
        title="DeepExtract — Personalized RAG Assistant API",
        description="Source-grounded document question answering API",
        version="0.1.0",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=f"{API_PREFIX}/redoc",
        openapi_url=f"{API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )
    register_exception_handlers(application)
    application.include_router(health.router)
    application.include_router(auth.router, prefix=API_PREFIX)
    application.include_router(users.router, prefix=API_PREFIX)
    application.include_router(threads.router, prefix=API_PREFIX)
    application.include_router(documents.router, prefix=API_PREFIX)
    application.include_router(data_sources.router, prefix=API_PREFIX)
    application.include_router(retrieval.router, prefix=API_PREFIX)
    application.include_router(chat.router, prefix=API_PREFIX)
    application.include_router(metrics.router, prefix=API_PREFIX)
    return application


app = create_app()
