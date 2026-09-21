"""Page-preserving deterministic text chunking."""

from backend.app.ingestion.normalization import content_hash, normalize_text
from backend.app.ingestion.types import ExtractedPage, IngestionChunk


def chunk_pages(
    pages: list[ExtractedPage], chunk_size: int, chunk_overlap: int
) -> list[IngestionChunk]:
    """Split normalized pages while suppressing duplicate chunks within a document."""
    chunks: list[IngestionChunk] = []
    seen_hashes: set[str] = set()
    for page in pages:
        normalized_page = normalize_text(page.text)
        if not normalized_page:
            continue
        start = 0
        while start < len(normalized_page):
            end = min(start + chunk_size, len(normalized_page))
            if end < len(normalized_page):
                boundary = normalized_page.rfind(" ", start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary
            text = normalized_page[start:end].strip()
            normalized_chunk = normalize_text(text)
            digest = content_hash(normalized_chunk)
            if normalized_chunk and digest not in seen_hashes:
                seen_hashes.add(digest)
                chunks.append(
                    IngestionChunk(
                        chunk_index=len(chunks),
                        page_number=page.page_number,
                        content=text,
                        normalized_content=normalized_chunk,
                        content_hash=digest,
                        token_count=len(normalized_chunk.split()),
                        metadata=(
                            page.metadata
                            if page.page_number is None
                            else {**page.metadata, "page_number": page.page_number}
                        ),
                    )
                )
            if end >= len(normalized_page):
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks
