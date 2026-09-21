"""Conversation history and metrics ownership tests."""

from httpx import AsyncClient

from backend.tests.conftest import LoginUser, RegisterUser
from backend.tests.integration.test_auth_and_threads import bearer


async def test_chat_history_and_metrics_are_persisted(
    client: AsyncClient, register_user: RegisterUser, login_user: LoginUser
) -> None:
    await register_user("alice", "alice@example.com")
    token = await login_user("alice@example.com")
    headers = bearer(token)
    thread = await client.post(
        "/api/v1/threads", json={"title": "Policy questions"}, headers=headers
    )
    thread_id = thread.json()["id"]

    answer = await client.post(
        f"/api/v1/chat/{thread_id}", json={"question": "Hello"}, headers=headers
    )
    history = await client.get(f"/api/v1/chat/{thread_id}/history", headers=headers)
    metrics = await client.get("/api/v1/metrics/overview", headers=headers)

    assert answer.status_code == 200
    assert answer.json()["grounded"] is False
    assert answer.json()["sources"] == []
    assert [message["role"] for message in history.json()] == ["user", "assistant"]
    assert metrics.json()["activity"]["threads"] == 1
    assert metrics.json()["activity"]["messages"] == 2
    assert metrics.json()["knowledge_base"]["indexed_chunks"] == 0


async def test_chat_stream_emits_tokens_then_persists_one_complete_turn(
    client: AsyncClient, register_user: RegisterUser, login_user: LoginUser
) -> None:
    await register_user("stream-user", "stream@example.com")
    headers = bearer(await login_user("stream@example.com"))
    thread = await client.post("/api/v1/threads", json={"title": "Streaming"}, headers=headers)
    thread_id = thread.json()["id"]

    response = await client.post(
        f"/api/v1/chat/{thread_id}/stream",
        json={"question": "Hello"},
        headers=headers,
    )
    history = await client.get(f"/api/v1/chat/{thread_id}/history", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in response.text
    assert 'event: complete\ndata: {"thread_id":' in response.text
    assert "Hello. Ask me a question" in response.text
    assert [message["role"] for message in history.json()] == ["user", "assistant"]
    assert history.json()[-1]["content"].startswith("Hello.")


async def test_chat_does_not_disclose_foreign_threads(
    client: AsyncClient, register_user: RegisterUser, login_user: LoginUser
) -> None:
    await register_user("alice", "alice@example.com")
    await register_user("bob", "bob@example.com")
    alice = bearer(await login_user("alice@example.com"))
    bob = bearer(await login_user("bob@example.com"))
    thread = await client.post("/api/v1/threads", json={"title": "Private"}, headers=alice)
    thread_id = thread.json()["id"]

    history = await client.get(f"/api/v1/chat/{thread_id}/history", headers=bob)
    answer = await client.post(f"/api/v1/chat/{thread_id}", json={"question": "Hello"}, headers=bob)

    assert history.status_code == 404
    assert answer.status_code == 404
