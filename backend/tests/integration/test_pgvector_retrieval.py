"""Real PostgreSQL/pgvector ingestion and tenant-isolation acceptance test."""

import os
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import text

from backend.app.core.config import Settings
from backend.app.db.session import Database
from backend.app.main import create_app
from backend.tests.conftest import TEST_PASSWORD
from backend.tests.integration.test_auth_and_threads import bearer


@pytest_asyncio.fixture
async def postgres_client() -> AsyncIterator[AsyncClient]:
    """Use migrated PostgreSQL when CI or a developer explicitly provides its URL."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url or not database_url.startswith("postgresql"):
        pytest.skip("DATABASE_URL does not point to PostgreSQL")
    settings = Settings(
        app_env="test",
        database_url_override=SecretStr(database_url),
        jwt_secret=SecretStr("postgres-test-secret-with-at-least-32-characters"),
        embedding_provider="fake",
        llm_provider="fake",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://postgres-test"
        ) as client:
            yield client


async def _register_and_login(client: AsyncClient, label: str) -> str:
    unique = uuid4().hex
    email = f"{label}-{unique}@example.com"
    registered = await client.post(
        "/api/v1/auth/register",
        json={"username": f"{label}-{unique[:12]}", "email": email, "password": TEST_PASSWORD},
    )
    assert registered.status_code == 201
    logged_in = await client.post(
        "/api/v1/auth/login", json={"email": email, "password": TEST_PASSWORD}
    )
    assert logged_in.status_code == 200
    return str(logged_in.json()["access_token"])


def _text_pdf(content: str) -> bytes:
    """Create a minimal one-page text PDF without a test-only PDF generator dependency."""
    escaped = content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 11 Tf 72 720 Td ({escaped}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, value in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{index} 0 obj\n".encode() + value + b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode())
    document.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(document)


async def test_pgvector_search_returns_sources_without_crossing_owners(
    postgres_client: AsyncClient,
) -> None:
    alice_token = await _register_and_login(postgres_client, "pg-alice")
    bob_token = await _register_and_login(postgres_client, "pg-bob")
    content = (
        "The employee leave policy provides twenty annual leave days each calendar year.\n"
        "Unused leave may carry forward up to five days with manager approval."
    )
    pdf = _text_pdf(content)
    thread = await postgres_client.post(
        "/api/v1/threads",
        json={"title": "Leave policy"},
        headers=bearer(alice_token),
    )
    thread_id = thread.json()["id"]
    uploaded = await postgres_client.post(
        "/api/v1/documents/upload",
        files={"file": ("employee-policy.pdf", pdf, "application/pdf")},
        headers=bearer(alice_token),
    )
    duplicate = await postgres_client.post(
        "/api/v1/documents/upload",
        files={"file": ("renamed-policy.pdf", pdf, "application/pdf")},
        headers=bearer(alice_token),
    )
    alice_search = await postgres_client.post(
        "/api/v1/retrieval/search",
        json={"query": content, "top_k": 5},
        headers=bearer(alice_token),
    )
    alice_thread_search = await postgres_client.post(
        "/api/v1/retrieval/search",
        json={"query": content, "top_k": 5, "thread_id": thread_id},
        headers=bearer(alice_token),
    )
    bob_search = await postgres_client.post(
        "/api/v1/retrieval/search",
        json={"query": content, "top_k": 5},
        headers=bearer(bob_token),
    )
    answer = await postgres_client.post(
        f"/api/v1/chat/{thread_id}",
        json={"question": "How many annual leave days does the employee leave policy provide?"},
        headers=bearer(alice_token),
    )
    metrics = await postgres_client.get("/api/v1/metrics/overview", headers=bearer(alice_token))
    metadata_search = await postgres_client.post(
        "/api/v1/retrieval/search",
        json={"query": content, "metadata": {"page_number": 1}},
        headers=bearer(alice_token),
    )
    unrelated = await postgres_client.post(
        f"/api/v1/chat/{thread_id}",
        json={"question": "Explain interstellar xylophone mineral taxonomy."},
        headers=bearer(alice_token),
    )
    stream = await postgres_client.post(
        f"/api/v1/chat/{thread_id}/stream",
        json={"question": "Hello"},
        headers=bearer(alice_token),
    )
    database_url = os.environ["DATABASE_URL"]
    database = Database(database_url)
    try:
        async with database.sessions() as session:
            any_thread_checkpoints = await session.scalar(
                text("SELECT count(*) FROM checkpoints WHERE thread_id LIKE :suffix"),
                {"suffix": f"%:{thread_id}"},
            )
            embedded_chunks = await session.scalar(
                text(
                    "SELECT count(*) FROM document_chunks "
                    "WHERE document_id = :document_id AND embedding IS NOT NULL"
                ),
                {"document_id": UUID(uploaded.json()["document"]["id"])},
            )
    finally:
        await database.close()

    assert thread.status_code == 201
    assert uploaded.status_code == 201
    assert alice_search.status_code == 200
    assert alice_search.json()["results"][0]["document_name"] == "employee-policy.pdf"
    assert alice_search.json()["results"][0]["page_number"] == 1
    assert alice_search.json()["results"][0]["score"] == pytest.approx(1.0)
    assert alice_thread_search.status_code == 200
    assert alice_thread_search.json()["results"][0]["document_name"] == "employee-policy.pdf"
    assert bob_search.status_code == 200
    assert bob_search.json()["results"] == []
    assert answer.status_code == 200
    assert answer.json()["grounded"] is True
    assert answer.json()["sources"][0]["document_name"] == "employee-policy.pdf"
    assert answer.json()["sources"][0]["page_number"] == 1
    assert "twenty annual leave days" in answer.json()["answer"]
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["document"]["id"] == uploaded.json()["document"]["id"]
    assert metadata_search.json()["results"][0]["page_number"] == 1
    assert unrelated.json()["grounded"] is False
    assert unrelated.json()["sources"] == []
    assert "could not find enough evidence" in unrelated.json()["answer"]
    assert "event: token" in stream.text
    assert "event: complete" in stream.text
    assert int(any_thread_checkpoints or 0) > 0
    assert int(embedded_chunks or 0) > 0
    assert metrics.json()["knowledge_base"]["indexed_chunks"] >= 1
