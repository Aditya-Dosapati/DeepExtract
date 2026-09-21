"""Typed values shared by document ingestion stages."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text and source metadata extracted from one logical page."""

    text: str
    page_number: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IngestionChunk:
    """Normalized, hashed chunk ready for embedding."""

    chunk_index: int
    page_number: int | None
    content: str
    normalized_content: str
    content_hash: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
