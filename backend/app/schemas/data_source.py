"""External data-source request and response schemas."""

from pydantic import BaseModel, Field


class DriveSyncResponse(BaseModel):
    """Counts returned by one manual Google Drive folder sync."""

    folder_id: str
    discovered: int = Field(ge=0)
    indexed: int = Field(ge=0)
    unchanged: int = Field(ge=0)
    duplicates: int = Field(ge=0)
    skipped: int = Field(ge=0)
    failed: int = Field(ge=0)
