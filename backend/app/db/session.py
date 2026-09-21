"""Async SQLAlchemy engine and request-scoped sessions."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class Database:
    """Own the async engine and session factory for one application instance."""

    def __init__(self, database_url: str) -> None:
        engine_options: dict[str, object] = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            engine_options.pop("pool_pre_ping")
        self.engine: AsyncEngine = create_async_engine(database_url, **engine_options)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def is_ready(self) -> bool:
        """Return whether the database accepts a trivial query."""
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception:
            return False
        return True

    async def close(self) -> None:
        """Dispose all pooled connections."""
        await self.engine.dispose()


def get_database(request: Request) -> Database:
    """Return the database owned by the current FastAPI application."""
    database: Database = request.app.state.database
    return database


async def get_session(
    database: Annotated[Database, Depends(get_database)],
) -> AsyncIterator[AsyncSession]:
    """Yield one transaction-aware session for an HTTP request."""
    async with database.sessions() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


SessionDep = Annotated[AsyncSession, Depends(get_session)]
DatabaseDep = Annotated[Database, Depends(get_database)]
