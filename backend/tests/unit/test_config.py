"""Settings validation tests."""

import pytest
from pydantic import SecretStr, ValidationError

from backend.app.core.config import Settings


def test_database_url_escapes_credentials() -> None:
    settings = Settings(
        postgres_user="rag user",
        postgres_password=SecretStr("p@ss/word"),
        database_url_override=None,
    )

    assert settings.database_url.startswith("postgresql+asyncpg://rag+user:p%40ss%2Fword@")


def test_empty_database_url_uses_postgres_components() -> None:
    settings = Settings(
        postgres_host="database.internal",
        database_url_override=SecretStr(""),
    )

    assert "@database.internal:5432/" in settings.database_url


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=500, chunk_overlap=500)


def test_default_jwt_secret_is_rejected_in_production() -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(
            app_env="production",
            jwt_secret=SecretStr("replace-with-secure-secret"),
        )


def test_mcp_path_is_normalized() -> None:
    assert Settings(mcp_path="/tools/").mcp_path == "/tools"


def test_default_postgres_host_is_localhost() -> None:
    settings = Settings()
    assert settings.postgres_host == "localhost"


def test_database_url_normalizes_render_and_standard_schemes() -> None:
    # Render postgres:// format
    settings_render = Settings(
        database_url_override=SecretStr(
            "postgres://user:pass@dpg-xxx.render.com:5432/db?sslmode=require"
        )
    )
    assert (
        settings_render.database_url
        == "postgresql+asyncpg://user:pass@dpg-xxx.render.com:5432/db?sslmode=require"
    )
    assert (
        settings_render.checkpoint_database_url
        == "postgresql://user:pass@dpg-xxx.render.com:5432/db?sslmode=require"
    )

    # Standard postgresql:// format
    settings_std = Settings(
        database_url_override=SecretStr("postgresql://user:pass@host:5432/db")
    )
    assert settings_std.database_url == "postgresql+asyncpg://user:pass@host:5432/db"
    assert settings_std.checkpoint_database_url == "postgresql://user:pass@host:5432/db"

    # SQLAlchemy asyncpg format
    settings_asyncpg = Settings(
        database_url_override=SecretStr("postgresql+asyncpg://user:pass@host:5432/db")
    )
    assert settings_asyncpg.database_url == "postgresql+asyncpg://user:pass@host:5432/db"
    assert settings_asyncpg.checkpoint_database_url == "postgresql://user:pass@host:5432/db"

    # SQLite format returns None for checkpoint URL (InMemorySaver)
    settings_sqlite = Settings(
        database_url_override=SecretStr("sqlite+aiosqlite:///test.db")
    )
    assert settings_sqlite.database_url == "sqlite+aiosqlite:///test.db"
    assert settings_sqlite.checkpoint_database_url is None

