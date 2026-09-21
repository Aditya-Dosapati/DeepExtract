"""Lifecycle management for LangGraph conversation checkpoints."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from backend.app.core.config import Settings

type AgentCheckpointer = BaseCheckpointSaver[Any]


def checkpoint_thread_id(owner_id: UUID, thread_id: UUID) -> str:
    """Return the tenant-safe identifier shared by graph invocation and cleanup."""
    return f"{owner_id}:{thread_id}"


@asynccontextmanager
async def create_checkpointer(settings: Settings) -> AsyncIterator[AgentCheckpointer]:
    """Use PostgreSQL in deployed environments and isolated memory in SQLite tests."""
    connection_url = settings.checkpoint_database_url
    if connection_url is None:
        yield InMemorySaver()
        return
    async with AsyncPostgresSaver.from_conn_string(connection_url) as saver:
        await saver.setup()
        yield saver
