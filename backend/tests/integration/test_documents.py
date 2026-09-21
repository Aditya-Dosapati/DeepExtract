"""Document ingestion, deduplication, and ownership API tests."""

from typing import Any

from httpx import AsyncClient

from backend.app.api import documents as documents_api
from backend.tests.conftest import LoginUser, RegisterUser
from backend.tests.integration.test_auth_and_threads import bearer


async def test_txt_upload_deduplication_and_owner_isolation(
    client: AsyncClient, register_user: RegisterUser, login_user: LoginUser
) -> None:
    await register_user("alice", "alice@example.com")
    await register_user("bob", "bob@example.com")
    alice_token = await login_user("alice@example.com")
    bob_token = await login_user("bob@example.com")
    upload = {"file": ("policy.txt", b"Employees receive twenty days of leave.", "text/plain")}

    created = await client.post(
        "/api/v1/documents/upload", files=upload, headers=bearer(alice_token)
    )
    duplicate = await client.post(
        "/api/v1/documents/upload", files=upload, headers=bearer(alice_token)
    )
    document_id = created.json()["document"]["id"]
    bob_list = await client.get("/api/v1/documents", headers=bearer(bob_token))
    bob_read = await client.get(f"/api/v1/documents/{document_id}", headers=bearer(bob_token))
    bob_delete = await client.delete(f"/api/v1/documents/{document_id}", headers=bearer(bob_token))
    alice_list = await client.get("/api/v1/documents", headers=bearer(alice_token))

    assert created.status_code == 201
    assert created.json()["duplicate"] is False
    assert created.json()["chunks_created"] == 1
    assert created.json()["document"]["status"] == "completed"
    assert duplicate.status_code == 201
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["chunks_created"] == 0
    assert duplicate.json()["document"]["id"] == document_id
    assert bob_list.json() == []
    assert bob_read.status_code == 404
    assert bob_delete.status_code == 404
    assert len(alice_list.json()) == 1


async def test_upload_thread_scope_must_be_owned(
    client: AsyncClient, register_user: RegisterUser, login_user: LoginUser
) -> None:
    await register_user("alice", "alice@example.com")
    await register_user("bob", "bob@example.com")
    alice_token = await login_user("alice@example.com")
    bob_token = await login_user("bob@example.com")
    thread = await client.post(
        "/api/v1/threads",
        json={"title": "Alice thread"},
        headers=bearer(alice_token),
    )

    response = await client.post(
        "/api/v1/documents/upload",
        data={"thread_id": thread.json()["id"]},
        files={"file": ("notes.txt", b"private notes", "text/plain")},
        headers=bearer(bob_token),
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "thread_not_found"


async def test_document_validation_errors_are_structured(
    client: AsyncClient, register_user: RegisterUser, login_user: LoginUser
) -> None:
    await register_user("alice", "alice@example.com")
    token = await login_user("alice@example.com")

    response = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("malware.exe", b"data", "application/octet-stream")},
        headers=bearer(token),
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_document"


async def test_embedding_failure_is_durable_and_visible_in_metrics(
    client: AsyncClient,
    register_user: RegisterUser,
    login_user: LoginUser,
    monkeypatch: Any,
) -> None:
    class FailingEmbeddingProvider:
        async def embed_documents(self, _texts: list[str]) -> list[list[float]]:
            raise RuntimeError("provider details must not leak")

    monkeypatch.setattr(
        documents_api,
        "create_embedding_provider",
        lambda _settings: FailingEmbeddingProvider(),
    )
    await register_user("alice", "alice@example.com")
    headers = bearer(await login_user("alice@example.com"))

    response = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("policy.txt", b"Index this policy", "text/plain")},
        headers=headers,
    )
    metrics = await client.get("/api/v1/metrics/overview", headers=headers)
    documents = await client.get("/api/v1/documents", headers=headers)

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "ingestion_unavailable",
        "message": "Document indexing failed",
        "details": None,
    }
    assert metrics.json()["activity"]["failed_ingestion_jobs"] == 1
    assert metrics.json()["knowledge_base"]["failed_documents"] == 1
    assert documents.json()[0]["status"] == "failed"


async def test_delete_document_and_cascaded_chunks_and_retrieval(
    client: AsyncClient, register_user: RegisterUser, login_user: LoginUser
) -> None:
    await register_user("charlie", "charlie@example.com")
    headers = bearer(await login_user("charlie@example.com"))

    upload = {
        "file": (
            "security_spec.txt",
            b"Acme proprietary security protocol requires token validation on every request.",
            "text/plain",
        )
    }
    created = await client.post("/api/v1/documents/upload", files=upload, headers=headers)
    assert created.status_code == 201
    document_id = created.json()["document"]["id"]

    # Verify document and chunks are indexed
    docs = await client.get("/api/v1/documents", headers=headers)
    assert len(docs.json()) == 1
    metrics_before = await client.get("/api/v1/metrics/overview", headers=headers)
    assert metrics_before.json()["knowledge_base"]["indexed_documents"] == 1
    assert metrics_before.json()["knowledge_base"]["indexed_chunks"] == 1

    # Perform retrieval before deletion
    search_before = await client.post(
        "/api/v1/retrieval/search",
        json={"query": "security protocol token validation", "top_k": 5},
        headers=headers,
    )
    assert search_before.status_code == 200
    assert len(search_before.json()["results"]) == 1

    # Delete the document
    deleted = await client.delete(f"/api/v1/documents/{document_id}", headers=headers)
    assert deleted.status_code == 204

    # Verify document list is empty
    docs_after = await client.get("/api/v1/documents", headers=headers)
    assert docs_after.json() == []

    # Verify get by ID returns 404
    get_after = await client.get(f"/api/v1/documents/{document_id}", headers=headers)
    assert get_after.status_code == 404

    # Verify metrics show 0 indexed documents and 0 chunks
    metrics_after = await client.get("/api/v1/metrics/overview", headers=headers)
    assert metrics_after.json()["knowledge_base"]["indexed_documents"] == 0
    assert metrics_after.json()["knowledge_base"]["indexed_chunks"] == 0

    # Verify retrieval after deletion returns 0 results
    search_after = await client.post(
        "/api/v1/retrieval/search",
        json={"query": "security protocol token validation", "top_k": 5},
        headers=headers,
    )
    assert search_after.status_code == 200
    assert len(search_after.json()["results"]) == 0

    # Verify deleting again returns 404
    delete_again = await client.delete(f"/api/v1/documents/{document_id}", headers=headers)
    assert delete_again.status_code == 404
    assert delete_again.json()["error"]["code"] == "document_not_found"

