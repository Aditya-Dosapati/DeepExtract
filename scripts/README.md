# Operational commands

Run commands from the repository root after `uv sync --frozen --extra frontend`.

## API smoke check

Start the stack, then verify registration, authentication, upload, retrieval, and chat:

```bash
uv run python -m scripts.api_smoke
```

Set `BACKEND_URL` only when the API is not available at `http://localhost:8000`.

## Local MCP smoke check

Obtain an application JWT from `POST /api/v1/auth/login`, then run:

```bash
MCP_ACCESS_TOKEN="<application-token>" uv run python -m scripts.mcp_smoke
```

Optional settings:

- `MCP_SERVER_URL`: local MCP URL; defaults to `http://localhost:8001/mcp`
- `MCP_SMOKE_QUERY`: also exercises `search_documents` when present

The script never prints the access token or authorization header.

## Full acceptance workflow

Run the live workflow against Docker Compose and then verify persistence after restart:

```bash
uv run python -m scripts.full_acceptance
docker compose restart
uv run python -m scripts.full_acceptance --verify-restart
```

This covers text-bearing PDF ingestion, citations, deduplication, owner isolation,
insufficient-evidence responses, streaming, local MCP search and answer tools, stored vectors,
and PostgreSQL checkpoint persistence. Temporary credentials are created with owner-only file
permissions and are not printed. Set `ACCEPTANCE_STATE_PATH` to choose a different secure state
location.

## Google Drive synchronization

Google Drive sync is an authenticated API operation rather than a local command-line script.
After configuring the `GOOGLE_DRIVE_*` variables and starting the stack:

```bash
curl -X POST http://localhost:8000/api/v1/data-sources/google-drive/sync \
  -H "Authorization: Bearer <application-token>"
```

The Google OAuth access token belongs only in the untracked `.env`; it is not an argument to
this request.

## Repository validation

```bash
make validate
docker compose config --quiet
```

`make validate` runs Ruff linting and formatting checks, strict mypy, and pytest. Compose
validation confirms that every configured service can be rendered without printing the
expanded configuration, which may contain local secret values.
