"""Persisted conversation message model."""

from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Enum, ForeignKey, Index, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.thread import ConversationThread
    from backend.app.models.user import User


class MessageRole(StrEnum):
    """Supported conversation message authors."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(TimestampMixin, Base):
    """A user-owned message belonging to one authorized conversation."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_thread_created", "thread_id", "created_at"),)

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    thread_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("conversation_threads.id", ondelete="CASCADE"),
        nullable=False,
    )
    owner_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, name="message_role", native_enum=False, length=20), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), default=list, nullable=False
    )

    thread: Mapped["ConversationThread"] = relationship(back_populates="messages")
    owner: Mapped["User"] = relationship(back_populates="messages")
