"""Authenticated operational dashboard metrics."""

from fastapi import APIRouter

from backend.app.auth.dependencies import CurrentUserDep
from backend.app.db.session import SessionDep
from backend.app.repositories.metrics import get_owner_metrics
from backend.app.schemas.metrics import ActivityMetrics, KnowledgeBaseMetrics, MetricsOverview

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/overview", response_model=MetricsOverview)
async def metrics_overview(user: CurrentUserDep, session: SessionDep) -> MetricsOverview:
    """Return current PostgreSQL inventory and activity for the authenticated owner."""
    metrics = await get_owner_metrics(session, user.id)
    return MetricsOverview(
        knowledge_base=KnowledgeBaseMetrics(
            documents=metrics.documents,
            indexed_documents=metrics.indexed_documents,
            processing_documents=metrics.processing_documents,
            failed_documents=metrics.failed_documents,
            indexed_chunks=metrics.indexed_chunks,
            stored_bytes=metrics.stored_bytes,
            last_indexed_at=metrics.last_indexed_at,
        ),
        activity=ActivityMetrics(
            threads=metrics.threads,
            messages=metrics.messages,
            active_ingestion_jobs=metrics.active_ingestion_jobs,
            failed_ingestion_jobs=metrics.failed_ingestion_jobs,
        ),
    )
