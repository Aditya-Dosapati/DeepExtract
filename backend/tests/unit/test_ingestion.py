"""Document validation, normalization, chunking, and embedding tests."""

import math
from io import BytesIO

import pytest
from docx import Document as DocxDocument
from pypdf import PdfWriter

from backend.app.ingestion.chunking import chunk_pages
from backend.app.ingestion.embeddings import DeterministicEmbeddingProvider, validate_embeddings
from backend.app.ingestion.extractors import (
    DocumentValidationError,
    extract_pages,
    validate_upload,
)
from backend.app.ingestion.normalization import content_hash, normalize_text
from backend.app.ingestion.types import ExtractedPage


def test_normalization_is_unicode_safe_and_deterministic() -> None:
    source = "\uff26\uff55\uff4c\uff4c\u0000  width\r\n\r\n\r\ntext\t here"

    normalized = normalize_text(source)

    assert normalized == "Full width\n\ntext here"
    assert content_hash(normalized) == content_hash(normalize_text(source))
    assert len(content_hash(normalized)) == 64


def test_txt_validation_and_extraction() -> None:
    data = b"Policy text"

    extension = validate_upload("policy.txt", "text/plain", data, 1024)
    pages = extract_pages(extension, data)

    assert extension == ".txt"
    assert pages == [ExtractedPage(text="Policy text", page_number=None)]


def test_docx_validation_and_extraction() -> None:
    document = DocxDocument()
    document.add_paragraph("Benefits policy")
    buffer = BytesIO()
    document.save(buffer)
    data = buffer.getvalue()

    extension = validate_upload(
        "policy.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        data,
        len(data),
    )
    pages = extract_pages(extension, data)

    assert pages == [ExtractedPage(text="Benefits policy", page_number=None)]


def test_password_protected_pdf_is_rejected() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    buffer = BytesIO()
    writer.write(buffer)

    with pytest.raises(DocumentValidationError, match="Password-protected"):
        extract_pages(".pdf", buffer.getvalue())


@pytest.mark.parametrize(
    ("filename", "mime_type", "data"),
    [
        ("policy.exe", "application/octet-stream", b"value"),
        ("policy.pdf", "text/plain", b"%PDF-value"),
        ("policy.pdf", "application/pdf", b"not-a-pdf"),
        ("policy.txt", "text/plain", b"contains\x00null"),
    ],
)
def test_invalid_uploads_are_rejected(filename: str, mime_type: str, data: bytes) -> None:
    with pytest.raises(DocumentValidationError):
        validate_upload(filename, mime_type, data, 1024)


def test_chunking_preserves_pages_overlap_and_suppresses_duplicates() -> None:
    pages = [
        ExtractedPage(text="alpha beta gamma delta epsilon", page_number=1),
        ExtractedPage(text="alpha beta gamma delta epsilon", page_number=2),
    ]

    chunks = chunk_pages(pages, chunk_size=18, chunk_overlap=5)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert all(chunk.metadata["page_number"] == 1 for chunk in chunks)
    assert len({chunk.content_hash for chunk in chunks}) == len(chunks)


async def test_deterministic_embeddings_are_repeatable_and_normalized() -> None:
    provider = DeterministicEmbeddingProvider(dimension=8)

    first = await provider.embed_documents(["policy", "benefits"])
    second = await provider.embed_documents(["policy", "benefits"])

    assert first == second
    assert first[0] != first[1]
    assert math.isclose(sum(value * value for value in first[0]), 1.0)
    validate_embeddings(first, expected_count=2, dimension=8)


def test_embedding_validation_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError, match="dimension"):
        validate_embeddings([[0.1, 0.2]], expected_count=1, dimension=3)
