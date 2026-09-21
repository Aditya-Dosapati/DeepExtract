"""Bounded LangGraph workflow for owner-scoped grounded answers."""

import re
from collections.abc import AsyncIterator
from typing import Any, Literal, TypedDict, cast
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agents.providers import AnswerProvider
from backend.app.agents.types import INSUFFICIENT_EVIDENCE, AgentStreamEvent, Evidence
from backend.app.core.config import Settings
from backend.app.ingestion.embeddings import EmbeddingProvider, validate_embeddings
from backend.app.models.message import Message
from backend.app.retrieval.search import search_chunks


class AgentState(TypedDict, total=False):
    """Mutable state passed between explicit answering stages."""

    query: str
    retrieval_query: str
    classification: Literal["conversation", "knowledge"]
    needs_retrieval: bool
    hits: list[Evidence]
    retry_count: int
    answer: str
    grounded: bool
    sources: list[dict[str, Any]]


def classify_query(query: str) -> Literal["conversation", "knowledge"]:
    """Keep greetings out of retrieval while routing substantive requests to the KB."""
    normalized = re.sub(r"[^a-z\s]", "", query.casefold()).strip()
    conversational = {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
    }
    return "conversation" if normalized in conversational else "knowledge"


def rewrite_retrieval_query(query: str) -> str:
    """Remove common conversational framing before the single bounded retry."""
    rewritten = re.sub(
        r"^(please\s+)?(can|could|would)\s+you\s+(tell|explain|show)\s+(me\s+)?",
        "",
        query.strip(),
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        r"^(what|where|when|who|how)\s+(is|are|was|were)\s+", "", rewritten, flags=re.IGNORECASE
    )
    return rewritten.strip(" ?.!") or query.strip()


def source_records(hits: list[Evidence]) -> list[dict[str, Any]]:
    """Build citations exclusively from database-returned chunks."""
    return [
        {
            "document_id": hit["document_id"],
            "document_name": hit["document_name"],
            "page_number": hit["page_number"],
            "chunk_id": hit["chunk_id"],
            "chunk_index": hit["chunk_index"],
            "score": round(hit["score"], 4),
            "excerpt": hit["content"][:600],
        }
        for hit in hits
    ]


def build_agent_workflow(
    *,
    session: AsyncSession,
    owner_id: UUID,
    thread_id: UUID,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    answer_provider: AnswerProvider,
    history: list[Message],
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    """Compile the retrieval, retry, generation, and citation-validation graph."""

    async def classify(state: AgentState) -> AgentState:
        return {"classification": classify_query(state["query"])}

    async def determine_retrieval(state: AgentState) -> AgentState:
        return {"needs_retrieval": state["classification"] == "knowledge"}

    def route_retrieval(state: AgentState) -> Literal["retrieve", "generate"]:
        return "retrieve" if state["needs_retrieval"] else "generate"

    async def retrieve(state: AgentState) -> AgentState:
        vectors = await embedding_provider.embed_documents([state["retrieval_query"]])
        validate_embeddings(vectors, 1, settings.embedding_dimension)
        search_hits = await search_chunks(
            session,
            owner_id=owner_id,
            query_vector=vectors[0],
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
            thread_id=thread_id,
            include_global=True,
        )
        hits: list[Evidence] = [
            {
                "document_id": str(hit.chunk.document_id),
                "document_name": hit.document_name,
                "page_number": hit.chunk.page_number,
                "chunk_id": str(hit.chunk.id),
                "chunk_index": hit.chunk.chunk_index,
                "score": hit.score,
                "content": hit.chunk.content,
            }
            for hit in search_hits
        ]
        return {"hits": hits}

    async def grade_context(state: AgentState) -> AgentState:
        return {"grounded": bool(state.get("hits"))}

    def route_context(state: AgentState) -> Literal["rewrite", "generate"]:
        if not state.get("hits") and state["retry_count"] < settings.agent_max_retrieval_retries:
            return "rewrite"
        return "generate"

    async def rewrite(state: AgentState) -> AgentState:
        return {
            "retrieval_query": rewrite_retrieval_query(state["query"]),
            "retry_count": state["retry_count"] + 1,
        }

    async def generate(state: AgentState) -> AgentState:
        writer = get_stream_writer()
        if state["classification"] == "conversation":
            answer = "Hello. Ask me a question about the documents in your knowledge base."
            for token in re.findall(r"\S+\s*", answer):
                writer({"event": "token", "content": token})
            return {
                "answer": answer,
                "grounded": False,
            }
        hits = state.get("hits", [])
        if not hits:
            for token in re.findall(r"\S+\s*", INSUFFICIENT_EVIDENCE):
                writer({"event": "token", "content": token})
            return {"answer": INSUFFICIENT_EVIDENCE, "grounded": False}
        answer_parts: list[str] = []
        async for token in answer_provider.stream(state["query"], hits, history):
            answer_parts.append(token)
            writer({"event": "token", "content": token})
        answer = "".join(answer_parts).strip()
        if not answer:
            answer = INSUFFICIENT_EVIDENCE
            return {"answer": answer, "grounded": False}
        if answer == INSUFFICIENT_EVIDENCE:
            return {"answer": answer, "grounded": False}
        return {"answer": answer, "grounded": True}

    async def validate_sources(state: AgentState) -> AgentState:
        hits = state.get("hits", []) if state.get("grounded") else []
        if state.get("grounded") and not hits:
            return {"answer": INSUFFICIENT_EVIDENCE, "grounded": False, "sources": []}
        return {"sources": source_records(hits)}

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("determine_retrieval", determine_retrieval)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_context", grade_context)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("validate_sources", validate_sources)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "determine_retrieval")
    graph.add_conditional_edges("determine_retrieval", route_retrieval)
    graph.add_edge("retrieve", "grade_context")
    graph.add_conditional_edges("grade_context", route_context)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "validate_sources")
    graph.add_edge("validate_sources", END)
    return graph.compile(checkpointer=checkpointer)


def agent_config(owner_id: UUID, thread_id: UUID) -> RunnableConfig:
    """Namespace checkpoints by both owner and conversation identifiers."""
    return {"configurable": {"thread_id": f"{owner_id}:{thread_id}"}}


def initial_state(query: str) -> AgentState:
    """Return a complete per-turn input so older checkpoint values cannot leak forward."""
    return {
        "query": query,
        "retrieval_query": query,
        "hits": [],
        "retry_count": 0,
        "grounded": False,
        "sources": [],
    }


async def run_agent(
    *,
    session: AsyncSession,
    owner_id: UUID,
    thread_id: UUID,
    query: str,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    answer_provider: AnswerProvider,
    history: list[Message],
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> AgentState:
    """Run one bounded agent turn and return its validated state."""
    workflow = build_agent_workflow(
        session=session,
        owner_id=owner_id,
        thread_id=thread_id,
        settings=settings,
        embedding_provider=embedding_provider,
        answer_provider=answer_provider,
        history=history,
        checkpointer=checkpointer,
    )
    result = await workflow.ainvoke(initial_state(query), config=agent_config(owner_id, thread_id))
    return cast(AgentState, result)


async def stream_agent(
    *,
    session: AsyncSession,
    owner_id: UUID,
    thread_id: UUID,
    query: str,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    answer_provider: AnswerProvider,
    history: list[Message],
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> AsyncIterator[AgentStreamEvent]:
    """Stream provider tokens and finish with the validated graph state."""
    workflow = build_agent_workflow(
        session=session,
        owner_id=owner_id,
        thread_id=thread_id,
        settings=settings,
        embedding_provider=embedding_provider,
        answer_provider=answer_provider,
        history=history,
        checkpointer=checkpointer,
    )
    final_state: AgentState | None = None
    async for part in workflow.astream(
        initial_state(query),
        config=agent_config(owner_id, thread_id),
        stream_mode=["custom", "values"],
        version="v2",
    ):
        if part["type"] == "custom":
            data = part["data"]
            if isinstance(data, dict) and data.get("event") == "token":
                yield {"event": "token", "data": str(data.get("content", ""))}
        elif part["type"] == "values":
            final_state = cast(AgentState, part["data"])
    if final_state is None:
        raise RuntimeError("Agent stream completed without a final state")
    yield {"event": "complete", "data": dict(final_state)}
