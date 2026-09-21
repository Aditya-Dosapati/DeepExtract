"""Container and orchestrator health endpoints."""

from fastapi import APIRouter, status

from backend.app.core.errors import ApplicationError
from backend.app.db.session import DatabaseDep
from backend.app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    """Confirm the process can serve requests."""
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse)
async def ready(database: DatabaseDep) -> HealthResponse:
    """Confirm required database infrastructure is reachable."""
    if not await database.is_ready():
        raise ApplicationError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "database_unavailable",
            "Database is not ready",
        )
    return HealthResponse(status="ok")
