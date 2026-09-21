"""Authenticated external data-source synchronization endpoints."""

from fastapi import APIRouter

from backend.app.auth.dependencies import CurrentUserDep, SettingsDep
from backend.app.connectors.google_drive import DriveConnectorError, open_google_drive_client
from backend.app.core.errors import ApplicationError
from backend.app.db.session import SessionDep
from backend.app.ingestion.embeddings import create_embedding_provider
from backend.app.schemas.data_source import DriveSyncResponse
from backend.app.services.google_drive_sync import sync_google_drive_folder

router = APIRouter(prefix="/data-sources", tags=["data-sources"])


@router.post("/google-drive/sync", response_model=DriveSyncResponse)
async def sync_google_drive(
    user: CurrentUserDep,
    settings: SettingsDep,
    session: SessionDep,
) -> DriveSyncResponse:
    """Synchronize the configured Drive folder into the user's knowledge base."""
    try:
        async with open_google_drive_client(settings) as drive:
            summary = await sync_google_drive_folder(
                session,
                drive,
                settings,
                create_embedding_provider(settings),
                user.id,
            )
    except DriveConnectorError as exc:
        raise ApplicationError(503, "drive_unavailable", str(exc)) from exc
    return DriveSyncResponse.model_validate(summary, from_attributes=True)
