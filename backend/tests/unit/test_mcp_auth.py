"""MCP JWT adapter tests."""

from uuid import uuid4

from pydantic import SecretStr

from backend.app.auth.security import create_access_token
from backend.app.core.config import Settings
from backend.app.mcp.auth import ApplicationTokenVerifier


async def test_mcp_token_verifier_accepts_only_application_access_tokens() -> None:
    settings = Settings(
        app_env="test",
        database_url_override=SecretStr("sqlite+aiosqlite://"),
        jwt_secret=SecretStr("mcp-test-secret-with-at-least-32-characters"),
    )
    user_id = uuid4()
    token, _ = create_access_token(user_id, settings)
    verifier = ApplicationTokenVerifier(settings)

    verified = await verifier.verify_token(token)
    rejected = await verifier.verify_token(f"{token}tampered")

    assert verified is not None
    assert verified.subject == str(user_id)
    assert verified.client_id == str(user_id)
    assert verified.scopes == ["user"]
    assert rejected is None
