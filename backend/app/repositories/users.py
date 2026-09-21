"""User persistence operations."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.user import User


async def get_user_by_id(session: AsyncSession, user_id: UUID) -> User | None:
    """Fetch one user by primary key."""
    return await session.get(User, user_id)


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    """Fetch one user by canonical email."""
    result = await session.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def username_or_email_exists(session: AsyncSession, username: str, email: str) -> bool:
    """Return whether either unique account identifier is already present."""
    statement = select(User.id).where((User.username == username) | (User.email == email)).limit(1)
    return (await session.execute(statement)).scalar_one_or_none() is not None


async def add_user(session: AsyncSession, *, username: str, email: str, password_hash: str) -> User:
    """Persist a new account in the caller's transaction."""
    user = User(username=username, email=email, password_hash=password_hash)
    session.add(user)
    await session.flush()
    await session.refresh(user)
    return user
