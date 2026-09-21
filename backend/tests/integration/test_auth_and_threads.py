"""Authentication and ownership isolation API tests."""

from httpx import AsyncClient

from backend.tests.conftest import TEST_PASSWORD, LoginUser, RegisterUser


def bearer(token: str) -> dict[str, str]:
    """Build an authorization header for test requests."""
    return {"Authorization": f"Bearer {token}"}


async def test_registration_login_and_current_user(
    client: AsyncClient, register_user: RegisterUser, login_user: LoginUser
) -> None:
    registered = await register_user("alice", "alice@example.com")
    duplicate = await register_user("alice", "alice@example.com")
    invalid_login = await client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": "incorrect-password"},
    )
    token = await login_user("alice@example.com")
    current = await client.get("/api/v1/users/me", headers=bearer(token))

    assert registered.status_code == 201
    assert "password" not in registered.text
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "account_exists"
    assert invalid_login.status_code == 401
    assert current.status_code == 200
    assert current.json()["email"] == "alice@example.com"


async def test_thread_ownership_isolation(
    client: AsyncClient, register_user: RegisterUser, login_user: LoginUser
) -> None:
    await register_user("alice", "alice@example.com")
    await register_user("bob", "bob@example.com")
    alice_token = await login_user("alice@example.com")
    bob_token = await login_user("bob@example.com")

    created = await client.post(
        "/api/v1/threads",
        json={"title": "Alice private conversation"},
        headers=bearer(alice_token),
    )
    thread_id = created.json()["id"]
    bob_list = await client.get("/api/v1/threads", headers=bearer(bob_token))
    bob_read = await client.get(f"/api/v1/threads/{thread_id}", headers=bearer(bob_token))
    bob_delete = await client.delete(f"/api/v1/threads/{thread_id}", headers=bearer(bob_token))
    alice_read = await client.get(f"/api/v1/threads/{thread_id}", headers=bearer(alice_token))

    assert created.status_code == 201
    assert bob_list.json() == []
    assert bob_read.status_code == 404
    assert bob_delete.status_code == 404
    assert alice_read.status_code == 200


async def test_health_probes(client: AsyncClient) -> None:
    live = await client.get("/health/live")
    ready = await client.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 200
    assert ready.json() == {"status": "ok"}


async def test_validation_errors_use_structured_envelope(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json={"username": "x", "email": "invalid", "password": TEST_PASSWORD},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
