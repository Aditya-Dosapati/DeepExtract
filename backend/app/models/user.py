"""User account model."""

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import Boolean, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.message import Message
    from backend.app.models.thread import ConversationThread


class User(TimestampMixin, Base):
    """Authenticated account that owns every private application resource."""

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    threads: Mapped[list["ConversationThread"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )
    messages: Mapped[list["Message"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )
