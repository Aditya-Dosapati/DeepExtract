"""Current-user endpoint."""

from fastapi import APIRouter

from backend.app.auth.dependencies import CurrentUserDep
from backend.app.schemas.user import UserPublic

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserPublic)
async def current_user(user: CurrentUserDep) -> UserPublic:
    """Return the account represented by the verified token."""
    return UserPublic.model_validate(user)
