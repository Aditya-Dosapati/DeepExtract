"""Deterministic tests for the gold-set evaluation calculations."""

from scripts.evaluate_rag import (
    ExpectedSource,
    GoldCase,
    citation_precision,
    parse_judge_scores,
    retrieval_scores,
)


def gold_case() -> GoldCase:
    """Return one case with two accepted chunks."""
    return GoldCase(
        id="policy",
        category="answerable",
        question="What is the policy?",
        expected_answer="The policy answer.",
        expected_sources=[ExpectedSource(document_name="policy.pdf", chunk_indices=[2, 3])],
    )


def test_retrieval_scores_use_exact_gold_chunk_rank() -> None:
    results = [
        {"document_name": "other.pdf", "chunk_index": 0},
        {"document_name": "policy.pdf", "chunk_index": 3},
        {"document_name": "policy.pdf", "chunk_index": 9},
    ]

    scores = retrieval_scores(gold_case(), results)

    assert scores == {
        "hit_rate": 1.0,
        "reciprocal_rank": 0.5,
        "chunk_precision": 1 / 3,
    }


def test_citation_precision_rejects_non_gold_chunks() -> None:
    sources = [
        {"document_name": "policy.pdf", "chunk_index": 2},
        {"document_name": "policy.pdf", "chunk_index": 8},
    ]

    assert citation_precision(gold_case(), sources) == 0.5


def test_judge_parser_accepts_json_fence() -> None:
    scores = parse_judge_scores(
        """```json
{"context_relevance": 1, "context_sufficiency": 0.9, "answer_relevance": 1,
 "answer_correctness": 0.8, "faithfulness": 1, "rationale": "Supported."}
```"""
    )

    assert scores.context_sufficiency == 0.9
