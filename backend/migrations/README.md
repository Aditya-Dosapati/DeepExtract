# Database migrations

Alembic owns the PostgreSQL and pgvector schema. Apply migrations after PostgreSQL is healthy
and before starting application traffic:

```bash
uv run alembic upgrade head
```

Docker Compose applies the current migration before starting the backend and local MCP server.
For a production rollout, run migrations once as a controlled release step instead of allowing
multiple replicas to attempt the upgrade concurrently.

## Revision history

| Revision | Purpose |
| --- | --- |
| `20260803_0001` | Initial users and conversation schema |
| `20260803_0002` | Documents, chunks, embeddings, and ingestion jobs |
| `20260804_0003` | LangGraph PostgreSQL checkpoint tables |
| `20260804_0004` | External source manifest for Google Drive synchronization |

Inspect the installed and available revisions with:

```bash
uv run alembic current
uv run alembic history
```

Create a reviewed migration after changing SQLAlchemy models:

```bash
uv run alembic revision --autogenerate -m "describe the schema change"
```

Review generated constraints, indexes, server defaults, and downgrade behavior before commit.
Do not edit a revision that has already been applied outside local development; add a new
forward migration instead. Never place credentials or environment-specific connection strings
inside a migration.
