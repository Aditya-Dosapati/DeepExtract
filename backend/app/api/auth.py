"""Registration and login endpoints."""

from fastapi import APIRouter, status
from sqlalchemy.exc import IntegrityError

from backend.app.auth.dependencies import SettingsDep
from backend.app.auth.security import create_access_token, hash_password, verify_password
from backend.app.core.errors import ApplicationError
from backend.app.db.session import SessionDep
from backend.app.repositories.users import (
    add_user,
    get_user_by_email,
    username_or_email_exists,
)
from backend.app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from backend.app.schemas.common import ErrorResponse
from backend.app.schemas.user import UserPublic

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post(
    "/register",
    response_model=UserPublic,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
)
async def register(payload: RegisterRequest, session: SessionDep) -> UserPublic:
    """Register an account without returning password material."""
    if await username_or_email_exists(session, payload.username, str(payload.email)):
        raise ApplicationError(409, "account_exists", "Username or email is already registered")
    try:
        user = await add_user(
            session,
            username=payload.username,
            email=str(payload.email),
            password_hash=hash_password(payload.password),
        )
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise ApplicationError(
            409, "account_exists", "Username or email is already registered"
        ) from exc
    return UserPublic.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    responses={401: {"model": ErrorResponse}},
)
async def login(payload: LoginRequest, session: SessionDep, settings: SettingsDep) -> TokenResponse:
    """Verify credentials and issue a time-limited access token."""
    user = await get_user_by_email(session, str(payload.email))
    if (
        user is None
        or not user.is_active
        or not verify_password(payload.password, user.password_hash)
    ):
        raise ApplicationError(401, "invalid_credentials", "Email or password is incorrect")
    token, expires_in = create_access_token(user.id, settings)
    return TokenResponse(access_token=token, expires_in=expires_in)
