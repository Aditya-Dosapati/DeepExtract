"""Unicode-safe document normalization and deterministic hashing."""

import hashlib
import re
import unicodedata

_INVALID_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Apply NFKC, remove invalid controls, and normalize whitespace."""
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = _INVALID_CONTROLS.sub("", normalized)
    normalized = _HORIZONTAL_SPACE.sub(" ", normalized)
    normalized = "\n".join(line.strip() for line in normalized.splitlines())
    return _BLANK_LINES.sub("\n\n", normalized).strip()


def content_hash(normalized_text: str) -> str:
    """Return the SHA-256 hash of already-normalized UTF-8 text."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
