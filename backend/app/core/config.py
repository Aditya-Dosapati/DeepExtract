"""Validated application configuration."""

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _normalize_async_database_url(url: str) -> str:
    """Normalize any database URL for the async SQLAlchemy asyncpg driver."""
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://"):]
    if url.startswith("postgresql+psycopg://"):
        return "postgresql+asyncpg://" + url[len("postgresql+psycopg://"):]
    if url.startswith("postgresql+psycopg2://"):
        return "postgresql+asyncpg://" + url[len("postgresql+psycopg2://"):]
    if url.startswith("sqlite://") and not url.startswith("sqlite+aiosqlite://"):
        return "sqlite+aiosqlite://" + url[len("sqlite://"):]
    return url


def _normalize_psycopg_database_url(url: str) -> str | None:
    """Normalize a database URL into a standard Psycopg connection string for AsyncPostgresSaver."""
    if url.startswith("sqlite"):
        return None
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + url[len("postgresql+asyncpg://"):]
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://"):]
    if url.startswith("postgresql+psycopg2://"):
        return "postgresql://" + url[len("postgresql+psycopg2://"):]
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


class Settings(BaseSettings):
    """Configuration loaded lazily from environment variables."""

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_user: str = "rag_user"
    postgres_password: SecretStr = SecretStr("change-me")
    postgres_db: str = "agentic_rag"
    database_url_override: SecretStr | None = Field(default=None, alias="DATABASE_URL")

    jwt_secret: SecretStr = SecretStr("local-development-secret-change-me-32")
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    access_token_expire_minutes: int = Field(default=60, ge=1, le=10080)

    llm_provider: str = "fake"
    llm_model: str = ""
    llm_api_key: SecretStr | None = None
    embedding_provider: str = "fake"
    embedding_model: str = ""
    embedding_api_key: SecretStr | None = None
    embedding_dimension: int = Field(default=384, ge=1, le=4096)

    retrieval_top_k: int = Field(default=5, ge=1, le=100)
    retrieval_score_threshold: float = Field(default=0.2, ge=0, le=1)
    retrieval_max_context_chars: int = Field(default=12000, ge=1000, le=100000)
    agent_max_retrieval_retries: int = Field(default=1, ge=0, le=3)
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    max_upload_size_mb: int = Field(default=25, ge=1, le=100)

    mcp_transport: Literal["streamable-http", "stdio"] = "streamable-http"
    mcp_host: str = "0.0.0.0"  # noqa: S104
    mcp_port: int = Field(default=8001, ge=1, le=65535)
    mcp_path: str = "/mcp"
    mcp_issuer_url: str = "http://localhost:8000"
    mcp_resource_server_url: str = "http://localhost:8001/mcp"

    google_drive_mcp_url: str = "https://drivemcp.googleapis.com/mcp/v1"
    google_drive_folder_id: str = ""
    google_drive_access_token: SecretStr | None = None
    google_drive_quota_project: str = ""
    google_drive_include_subfolders: bool = True
    google_drive_max_files: int = Field(default=100, ge=1, le=1000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Normalize and validate the configured Python log level."""
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            msg = f"Unsupported LOG_LEVEL: {value}"
            raise ValueError(msg)
        return normalized

    @field_validator("mcp_path")
    @classmethod
    def validate_mcp_path(cls, value: str) -> str:
        """Require MCP mount paths to be absolute URL paths."""
        if not value.startswith("/") or value == "/":
            msg = "MCP_PATH must start with '/' and cannot be the root path"
            raise ValueError(msg)
        return value.rstrip("/")

    @field_validator("database_url_override", mode="before")
    @classmethod
    def normalize_database_url(cls, value: object) -> object:
        """Treat an empty optional DATABASE_URL as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, SecretStr) and not value.get_secret_value().strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_related_values(self) -> "Settings":
        """Validate settings whose constraints depend on other settings."""
        if self.chunk_overlap >= self.chunk_size:
            msg = "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
            raise ValueError(msg)
        if self.embedding_dimension not in {384, 1536} and self.embedding_dimension < 1:
            msg = "EMBEDDING_DIMENSION must be a valid positive integer dimension"
            raise ValueError(msg)
        insecure_secret = self.jwt_secret.get_secret_value() in {
            "replace-with-secure-secret",
            "local-development-secret-change-me-32",
        }
        if self.app_env == "production" and insecure_secret:
            msg = "JWT_SECRET must be changed in production"
            raise ValueError(msg)
        return self

    @property
    def database_url(self) -> str:
        """Return an async SQLAlchemy database URL without logging credentials."""
        if self.database_url_override is not None:
            return _normalize_async_database_url(self.database_url_override.get_secret_value())
        password = quote_plus(self.postgres_password.get_secret_value())
        user = quote_plus(self.postgres_user)
        return (
            f"postgresql+asyncpg://{user}:{password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def checkpoint_database_url(self) -> str | None:
        """Return a Psycopg-compatible PostgreSQL URL, or none for isolated SQLite tests."""
        if self.database_url_override is not None:
            return _normalize_psycopg_database_url(
                self.database_url_override.get_secret_value()
            )
        return _normalize_psycopg_database_url(self.database_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once when requested by the application."""
    return Settings()
