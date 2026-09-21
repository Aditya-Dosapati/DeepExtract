"""Password hashing and JWT access-token helpers."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from uuid import UUID

import jwt
from pwdlib import PasswordHash

from backend.app.core.config import Settings


class InvalidTokenError(ValueError):
    """Raised when an access token is invalid or has the wrong purpose."""


@dataclass(frozen=True, slots=True)
class AccessTokenClaims:
    """Verified claims used for authorization."""

    user_id: UUID
    expires_at: datetime


@lru_cache(maxsize=1)
def get_password_hasher() -> PasswordHash:
    """Create the recommended Argon2 password hasher only when first used."""
    return PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Hash a password using the current recommended algorithm."""
    return get_password_hasher().hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Safely verify a plaintext password."""
    return get_password_hasher().verify(password, password_hash)


def create_access_token(user_id: UUID, settings: Settings) -> tuple[str, int]:
    """Create a short-lived JWT containing only authorization claims."""
    issued_at = datetime.now(UTC)
    expires_in = settings.access_token_expire_minutes * 60
    expires_at = issued_at + timedelta(seconds=expires_in)
    payload = {
        "sub": str(user_id),
        "type": "access",
        "iat": issued_at,
        "exp": expires_at,
    }
    encoded = jwt.encode(
        payload,
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return encoded, expires_in


def decode_access_token(token: str, settings: Settings) -> AccessTokenClaims:
    """Verify an access JWT and return its typed authorization claims."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "type", "iat", "exp"]},
        )
        if payload["type"] != "access":
            raise InvalidTokenError
        user_id = UUID(payload["sub"])
        expires_at = datetime.fromtimestamp(payload["exp"], tz=UTC)
    except (jwt.PyJWTError, KeyError, TypeError, ValueError) as exc:
        raise InvalidTokenError from exc
    return AccessTokenClaims(user_id=user_id, expires_at=expires_at)
