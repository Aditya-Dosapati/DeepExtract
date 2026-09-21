.PHONY: lint format-check typecheck test validate

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy backend frontend scripts

test:
	uv run pytest --cov=backend.app --cov-report=term-missing --cov-fail-under=80

validate: lint format-check typecheck test
