FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app

COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY backend ./backend
COPY frontend ./frontend

RUN uv sync --frozen --no-dev --extra frontend \
    && groupadd --system app \
    && useradd --system --gid app --create-home app \
    && chown -R app:app /app

USER app

EXPOSE 8501

CMD [".venv/bin/streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
