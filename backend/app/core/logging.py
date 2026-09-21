"""Application logging configuration with secret-safe defaults."""

import logging


def configure_logging(level: str) -> None:
    """Configure standard output logging without request credential data."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
