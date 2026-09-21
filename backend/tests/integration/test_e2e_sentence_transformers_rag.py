"""Integration test for end-to-end RAG with Sentence Transformers local embeddings."""

from collections.abc import AsyncIterator
from uuid import UUID

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.main import create_app
from backend.tests.conftest import TEST_PASSWORD
from backend.tests.integration.test_auth_and_threads import bearer


@pytest_asyncio.fixture
async def sentence_transformers_client() -> AsyncIterator[AsyncClient]:
    """Test client running with Sentence Transformers local embeddings."""
    settings = Settings(
        app_env="test",
        database_url_override=SecretStr("sqlite+aiosqlite://"),
        jwt_secret=SecretStr("sentence-transformers-test-secret-with-at-least-32-chars"),
        embedding_provider="sentence_transformers",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
        llm_provider="fake",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with app.state.database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://sentence-transformers-test"
        ) as test_client:
            yield test_client


async def test_end_to_end_rag_with_sentence_transformers(
    sentence_transformers_client: AsyncClient,
) -> None:
    client = sentence_transformers_client

    # 1. Register & login
    email = "researcher@example.com"
    registered = await client.post(
        "/api/v1/auth/register",
        json={"username": "researcher", "email": email, "password": TEST_PASSWORD},
    )
    assert registered.status_code == 201

    logged_in = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": TEST_PASSWORD},
    )
    assert logged_in.status_code == 200
    token = logged_in.json()["access_token"]
    headers = bearer(token)

    # 2. Create conversation thread
    thread = await client.post(
        "/api/v1/threads",
        json={"title": "Policy Consultation"},
        headers=headers,
    )
    assert thread.status_code == 201
    thread_id = UUID(thread.json()["id"])

    # 3. Upload document
    document_text = (
        b"The corporate travel policy states that business class travel is permitted "
        b"only for flights longer than six continuous hours with prior VP approval.\n\n"
        b"Meals are reimbursed up to fifty dollars per day per traveler."
    )

    upload_resp = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("travel_policy.txt", document_text, "text/plain")},
        headers=headers,
    )
    assert upload_resp.status_code == 201
    doc_data = upload_resp.json()
    assert doc_data["duplicate"] is False
    assert doc_data["chunks_created"] >= 1

    # 4. Perform vector search
    search_resp = await client.post(
        "/api/v1/retrieval/search",
        json={"query": "What is the requirement for business class flights?", "top_k": 3},
        headers=headers,
    )
    assert search_resp.status_code == 200
    search_results = search_resp.json()["results"]
    assert len(search_results) >= 1
    assert "six continuous hours" in search_results[0]["excerpt"]
    assert search_results[0]["score"] > 0.5

    # 5. Ask question in thread
    chat_resp = await client.post(
        f"/api/v1/chat/{thread_id}",
        json={"question": "When can employees book business class flights?"},
        headers=headers,
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert chat_data["grounded"] is True
    assert len(chat_data["sources"]) >= 1
    assert "business class travel is permitted" in chat_data["answer"]

    # 6. Greeting query routes conversationally without knowledge retrieval
    greeting_resp = await client.post(
        f"/api/v1/chat/{thread_id}",
        json={"question": "Hello!"},
        headers=headers,
    )
    assert greeting_resp.status_code == 200
    greeting_data = greeting_resp.json()
    assert greeting_data["grounded"] is False
    assert greeting_data["sources"] == []
    assert "Hello" in greeting_data["answer"]
