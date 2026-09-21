"""Authentication request and response schemas."""

from typing import Annotated

from pydantic import BaseModel, EmailStr, Field, field_validator

Password = Annotated[str, Field(min_length=8, max_length=128)]


class RegisterRequest(BaseModel):
    """New account registration payload."""

    username: Annotated[str, Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")]
    email: EmailStr
    password: Password

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        """Store canonical lowercase usernames for reliable uniqueness."""
        return value.strip().lower()

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        """Store canonical lowercase email addresses."""
        return str(value).strip().lower()


class LoginRequest(BaseModel):
    """Email and password credentials."""

    email: EmailStr
    password: Password

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        """Match registration email normalization."""
        return str(value).strip().lower()


class TokenResponse(BaseModel):
    """Bearer access token returned after successful login."""

    access_token: str
    token_type: str = "bearer"  # noqa: S105
    expires_in: int
