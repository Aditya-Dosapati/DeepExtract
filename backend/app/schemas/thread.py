"""Conversation thread schemas."""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ThreadCreate(BaseModel):
    """Optional title for a new conversation."""

    title: Annotated[str, Field(min_length=1, max_length=200)] = "New conversation"


class ThreadPublic(BaseModel):
    """Authorized thread representation."""

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
