"""FastAPI dependency for the application-owned LangGraph checkpointer."""

from typing import Annotated

from fastapi import Depends, Request

from backend.app.agents.checkpoints import AgentCheckpointer


def get_checkpointer(request: Request) -> AgentCheckpointer:
    """Return the checkpointer initialized during application startup."""
    checkpointer: AgentCheckpointer = request.app.state.checkpointer
    return checkpointer


CheckpointerDep = Annotated[AgentCheckpointer, Depends(get_checkpointer)]
