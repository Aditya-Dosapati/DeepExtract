FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE alembic.ini ./
COPY backend ./backend

RUN uv sync --frozen --no-dev \
    && groupadd --system app \
    && useradd --system --gid app --create-home app \
    && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["sh", "-c", ".venv/bin/alembic upgrade head && exec .venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
