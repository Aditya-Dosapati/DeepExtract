"""Conversation thread model."""

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, Index, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.message import Message
    from backend.app.models.user import User


class ConversationThread(TimestampMixin, Base):
    """A user-owned conversation and document scope."""

    __tablename__ = "conversation_threads"
    __table_args__ = (Index("ix_conversation_threads_owner_created", "owner_id", "created_at"),)

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), default="New conversation", nullable=False)

    owner: Mapped["User"] = relationship(back_populates="threads")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="thread", cascade="all, delete-orphan", passive_deletes=True
    )
