"""Shared HTTP response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    """Stable machine-readable error details."""

    code: str
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    """Envelope returned for every handled API error."""

    error: ErrorBody


class HealthResponse(BaseModel):
    """Health probe result."""

    status: str = Field(examples=["ok"])
