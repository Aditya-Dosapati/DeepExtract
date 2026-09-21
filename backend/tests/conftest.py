"""Shared test application and authenticated-client fixtures."""

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest_asyncio
from httpx import ASGITransport, AsyncClient, Response
from pydantic import SecretStr

from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.main import create_app

type RegisterUser = Callable[[str, str], Awaitable[Response]]
type LoginUser = Callable[[str], Awaitable[str]]

TEST_PASSWORD = "Correct-Horse-Battery-7"


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """Run the API against an isolated in-memory async database."""
    settings = Settings(
        app_env="test",
        database_url_override=SecretStr("sqlite+aiosqlite://"),
        jwt_secret=SecretStr("unit-test-secret-with-at-least-32-characters"),
        embedding_provider="fake",
        llm_provider="fake",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with app.state.database.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as test_client:
            yield test_client


@pytest_asyncio.fixture
async def register_user(client: AsyncClient) -> RegisterUser:
    """Return a helper that registers one user."""

    async def register(username: str, email: str) -> Response:
        return await client.post(
            "/api/v1/auth/register",
            json={"username": username, "email": email, "password": TEST_PASSWORD},
        )

    return register


@pytest_asyncio.fixture
async def login_user(client: AsyncClient) -> LoginUser:
    """Return a helper that logs in and extracts a bearer token."""

    async def login(email: str) -> str:
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": TEST_PASSWORD},
        )
        assert response.status_code == 200
        return str(response.json()["access_token"])

    return login
