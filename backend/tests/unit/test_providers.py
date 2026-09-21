"""Unit tests for embedding and LLM providers including Groq and Sentence Transformers."""

import math
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from backend.app.agents.providers import (
    ExtractiveAnswerProvider,
    GroqAnswerProvider,
    OpenAIAnswerProvider,
    create_answer_provider,
)
from backend.app.agents.types import Evidence
from backend.app.core.config import Settings
from backend.app.ingestion.embeddings import (
    DeterministicEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformersEmbeddingProvider,
    create_embedding_provider,
    validate_embeddings,
)


def sample_evidence() -> list[Evidence]:
    return [
        {
            "document_id": "doc-1",
            "document_name": "policy.pdf",
            "page_number": 1,
            "chunk_id": "chunk-1",
            "chunk_index": 0,
            "score": 0.95,
            "content": "Employees are entitled to 20 days of annual paid leave.",
        }
    ]


# ---------------------------------------------------------
# Embedding Provider Tests
# ---------------------------------------------------------


def test_create_embedding_provider_fake() -> None:
    settings = Settings(embedding_provider="fake", embedding_dimension=384)
    provider = create_embedding_provider(settings)
    assert isinstance(provider, DeterministicEmbeddingProvider)
    assert provider.dimension == 384


def test_create_embedding_provider_openai_missing_key() -> None:
    settings = Settings(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_api_key=None,
    )
    with pytest.raises(ValueError, match="OpenAI embedding model and API key are required"):
        create_embedding_provider(settings)


def test_create_embedding_provider_openai_success() -> None:
    settings = Settings(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_api_key=SecretStr("sk-test-key"),
        embedding_dimension=1536,
    )
    provider = create_embedding_provider(settings)
    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert provider.dimension == 1536
    assert provider.model == "text-embedding-3-small"


def test_create_embedding_provider_sentence_transformers() -> None:
    settings = Settings(
        embedding_provider="sentence_transformers",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimension=384,
    )
    provider = create_embedding_provider(settings)
    assert isinstance(provider, SentenceTransformersEmbeddingProvider)
    assert provider.dimension == 384
    assert provider.model_name == "BAAI/bge-small-en-v1.5"


def test_create_embedding_provider_unsupported() -> None:
    settings = Settings(embedding_provider="unsupported_provider")
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        create_embedding_provider(settings)


async def test_sentence_transformers_embedding_generation() -> None:
    provider = SentenceTransformersEmbeddingProvider(
        model_name="BAAI/bge-small-en-v1.5", dimension=384
    )
    texts = [
        "Company vacation and holiday policy.",
        "How many vacation days do employees receive?",
    ]
    vectors = await provider.embed_documents(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384

    # Verify L2 normalization
    norm0 = math.sqrt(sum(x * x for x in vectors[0]))
    norm1 = math.sqrt(sum(x * x for x in vectors[1]))
    assert math.isclose(norm0, 1.0, rel_tol=1e-3)
    assert math.isclose(norm1, 1.0, rel_tol=1e-3)

    # Empty list handling
    assert await provider.embed_documents([]) == []


def test_embedding_dimension_validation() -> None:
    # Proper dimension passes
    validate_embeddings([[0.1] * 384], expected_count=1, dimension=384)

    # Count mismatch fails
    with pytest.raises(ValueError, match="wrong number of vectors"):
        validate_embeddings([[0.1] * 384], expected_count=2, dimension=384)

    # Dimension mismatch fails
    with pytest.raises(ValueError, match="unexpected dimension"):
        validate_embeddings([[0.1] * 1536], expected_count=1, dimension=384)

    # Non-finite values fail
    with pytest.raises(ValueError, match="non-finite value"):
        validate_embeddings([[float("nan")] * 384], expected_count=1, dimension=384)


# ---------------------------------------------------------
# LLM Provider Tests
# ---------------------------------------------------------


def test_create_answer_provider_fake() -> None:
    settings = Settings(llm_provider="fake")
    provider = create_answer_provider(settings)
    assert isinstance(provider, ExtractiveAnswerProvider)


def test_create_answer_provider_openai_missing_key() -> None:
    settings = Settings(llm_provider="openai", llm_model="gpt-4o", llm_api_key=None)
    with pytest.raises(ValueError, match="OpenAI chat model and API key are required"):
        create_answer_provider(settings)


def test_create_answer_provider_openai_success() -> None:
    settings = Settings(
        llm_provider="openai",
        llm_model="gpt-4o",
        llm_api_key=SecretStr("sk-test-key"),
    )
    provider = create_answer_provider(settings)
    assert isinstance(provider, OpenAIAnswerProvider)
    assert provider.model == "gpt-4o"


def test_create_answer_provider_groq_missing_key() -> None:
    settings = Settings(
        llm_provider="groq",
        llm_model="llama-3.3-70b-versatile",
        llm_api_key=None,
    )
    with pytest.raises(ValueError, match="Groq chat model and API key are required"):
        create_answer_provider(settings)


def test_create_answer_provider_groq_success() -> None:
    settings = Settings(
        llm_provider="groq",
        llm_model="llama-3.3-70b-versatile",
        llm_api_key=SecretStr("gsk-test-key"),
    )
    provider = create_answer_provider(settings)
    assert isinstance(provider, GroqAnswerProvider)
    assert provider.model == "llama-3.3-70b-versatile"


def test_create_answer_provider_unsupported() -> None:
    settings = Settings(llm_provider="unknown_provider")
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_answer_provider(settings)


async def test_groq_answer_provider_streaming_mock() -> None:
    provider = GroqAnswerProvider(api_key="gsk-test-key", model="llama-3.3-70b-versatile")

    # Mock chunk event
    class MockDelta:
        def __init__(self, content: str | None) -> None:
            self.content = content

    class MockChoice:
        def __init__(self, delta_content: str | None) -> None:
            self.delta = MockDelta(delta_content)

    class MockChunk:
        def __init__(self, content: str) -> None:
            self.choices = [MockChoice(content)]

    async def mock_stream_iterator() -> Any:
        for chunk in [MockChunk("Employees "), MockChunk("get "), MockChunk("20 days.")]:
            yield chunk

    with patch.object(
        provider.client.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_stream_iterator()),
    ):
        tokens = []
        async for token in provider.stream(
            query="How many days?",
            evidence=sample_evidence(),
            history=[],
        ):
            tokens.append(token)

        assert tokens == ["Employees ", "get ", "20 days."]
