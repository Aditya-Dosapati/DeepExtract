"""Serializable agent evidence and stream event types."""

from typing import Any, Literal, TypedDict

INSUFFICIENT_EVIDENCE = (
    "I could not find enough evidence in your indexed knowledge base to answer that. "
    "Try rephrasing the question or upload a relevant document."
)


class Evidence(TypedDict):
    """One authorized retrieval result safe for checkpoints and providers."""

    document_id: str
    document_name: str
    page_number: int | None
    chunk_id: str
    chunk_index: int
    score: float
    content: str


class AgentStreamEvent(TypedDict):
    """Internal event emitted while a graph turn is running."""

    event: Literal["token", "complete"]
    data: str | dict[str, Any]
