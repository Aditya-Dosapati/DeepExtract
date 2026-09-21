"""Backend package integrity checks."""

from backend import app


def test_backend_package_is_importable() -> None:
    """The backend package can be imported before features are added."""
    assert app.__doc__ == "Agentic RAG backend application package."
