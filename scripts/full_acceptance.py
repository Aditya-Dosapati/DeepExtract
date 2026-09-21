"""Exercise the complete live API, PostgreSQL, streaming, and MCP workflow."""

import argparse
import asyncio
import json
import os
import tempfile
from typing import Any
from uuid import UUID, uuid4

import httpx
from anyio import Path
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from psycopg import AsyncConnection

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/mcp")
DATABASE_URL = os.getenv("DATABASE_URL") or (
    "postgresql://"
    f"{os.getenv('POSTGRES_USER', 'rag_user')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'change-me')}@localhost:"
    f"{os.getenv('POSTGRES_PORT_FORWARD', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'agentic_rag')}"
)
DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
STATE_PATH = Path(
    os.getenv(
        "ACCEPTANCE_STATE_PATH",
        os.path.join(tempfile.gettempdir(), "agentic-rag-acceptance.json"),
    )
)


def text_pdf(content: str) -> bytes:
    """Create a minimal one-page PDF containing extractable text."""
    escaped = content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 11 Tf 72 720 Td ({escaped}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, value in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{index} 0 obj\n".encode() + value + b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode())
    document.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(document)


async def authenticate(client: httpx.AsyncClient, label: str) -> tuple[str, str, str]:
    """Create a unique account and return its email, password, and token."""
    suffix = uuid4().hex[:12]
    email = f"acceptance-{label}-{suffix}@example.com"
    password = f"Acceptance-{uuid4().hex}-A7!"
    response = await client.post(
        "/api/v1/auth/register",
        json={"username": f"{label}-{suffix}", "email": email, "password": password},
    )
    response.raise_for_status()
    response = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    response.raise_for_status()
    return email, password, str(response.json()["access_token"])


def auth(token: str) -> dict[str, str]:
    """Build a bearer header without logging its value."""
    return {"Authorization": f"Bearer {token}"}


async def mcp_checks(token: str, thread_id: str, query: str) -> dict[str, bool]:
    """Call search and answer through a real authenticated MCP session."""
    async with httpx.AsyncClient(headers=auth(token), timeout=30) as http_client:
        async with streamable_http_client(
            MCP_URL, http_client=http_client, terminate_on_close=False
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                searched = await session.call_tool("search_documents", {"query": query})
                answered = await session.call_tool(
                    "answer_from_documents", {"query": query, "thread_id": thread_id}
                )
    return {
        "mcp_search": not searched.isError
        and bool((searched.structuredContent or {}).get("results")),
        "mcp_answer": not answered.isError
        and bool((answered.structuredContent or {}).get("sources")),
    }


async def database_counts(document_id: str, thread_id: str) -> tuple[int, int]:
    """Confirm vectors and checkpoint rows are physically present in PostgreSQL."""
    async with await AsyncConnection.connect(DATABASE_URL) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT count(*) FROM document_chunks "
                "WHERE document_id = %s AND embedding IS NOT NULL",
                (UUID(document_id),),
            )
            chunks = int((await cursor.fetchone() or (0,))[0])
            await cursor.execute(
                "SELECT count(*) FROM checkpoints WHERE thread_id LIKE %s",
                (f"%:{thread_id}",),
            )
            checkpoints = int((await cursor.fetchone() or (0,))[0])
    return chunks, checkpoints


async def run_acceptance() -> None:
    """Create data and verify every externally visible acceptance boundary."""
    content = (
        "The Northstar leave policy grants employees twenty annual leave days. "
        "Unused leave may carry forward for five days with manager approval."
    )
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=45) as client:
        (alice_email, alice_password, alice), (_, _, bob) = await asyncio.gather(
            authenticate(client, "alice"), authenticate(client, "bob")
        )
        thread_response = await client.post(
            "/api/v1/threads", json={"title": "Acceptance policy"}, headers=auth(alice)
        )
        thread_response.raise_for_status()
        thread_id = str(thread_response.json()["id"])
        upload = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("northstar-policy.pdf", text_pdf(content), "application/pdf")},
            headers=auth(alice),
        )
        upload.raise_for_status()
        uploaded = upload.json()
        document_id = str(uploaded["document"]["id"])
        duplicate = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("renamed.pdf", text_pdf(content), "application/pdf")},
            headers=auth(alice),
        )
        duplicate.raise_for_status()
        alice_search = await client.post(
            "/api/v1/retrieval/search", json={"query": content}, headers=auth(alice)
        )
        bob_search = await client.post(
            "/api/v1/retrieval/search", json={"query": content}, headers=auth(bob)
        )
        answer = await client.post(
            f"/api/v1/chat/{thread_id}",
            json={"question": "How many annual leave days are granted?"},
            headers=auth(alice),
        )
        unrelated = await client.post(
            f"/api/v1/chat/{thread_id}",
            json={"question": "Explain interstellar xylophone mineral taxonomy."},
            headers=auth(alice),
        )
        streamed = await client.post(
            f"/api/v1/chat/{thread_id}/stream",
            json={"question": "Hello"},
            headers=auth(alice),
        )
        metrics = await client.get("/api/v1/metrics/overview", headers=auth(alice))
        for response in (alice_search, bob_search, answer, unrelated, streamed, metrics):
            response.raise_for_status()

    sources = answer.json()["sources"]
    checks: dict[str, bool] = {
        "pdf_page_citation": bool(sources) and sources[0]["page_number"] == 1,
        "duplicate_detection": duplicate.json()["duplicate"] is True,
        "owner_isolation": bob_search.json()["results"] == [],
        "insufficient_evidence": unrelated.json()["grounded"] is False,
        "sse_stream": "event: token" in streamed.text and "event: complete" in streamed.text,
        "metrics": metrics.json()["knowledge_base"]["indexed_chunks"] > 0,
        "vector_search": bool(alice_search.json()["results"]),
    }
    checks.update(await mcp_checks(alice, thread_id, content))
    chunks, checkpoints = await database_counts(document_id, thread_id)
    checks["stored_embeddings"] = chunks > 0
    checks["stored_checkpoints"] = checkpoints > 0
    if not all(checks.values()):
        raise SystemExit(f"Acceptance failed: {json.dumps(checks, sort_keys=True)}")

    state: dict[str, Any] = {
        "email": alice_email,
        "password": alice_password,
        "thread_id": thread_id,
        "document_id": document_id,
    }
    await STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    await STATE_PATH.chmod(0o600)
    print(json.dumps(checks, sort_keys=True))
    print(f"Restart state saved securely to {STATE_PATH}")


async def verify_restart() -> None:
    """Verify data created by run_acceptance remains after service restart."""
    state = json.loads(await STATE_PATH.read_text(encoding="utf-8"))
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=30) as client:
        response = await client.post(
            "/api/v1/auth/login",
            json={"email": state["email"], "password": state["password"]},
        )
        response.raise_for_status()
        token = str(response.json()["access_token"])
        documents = await client.get("/api/v1/documents", headers=auth(token))
        history = await client.get(
            f"/api/v1/chat/{state['thread_id']}/history", headers=auth(token)
        )
        documents.raise_for_status()
        history.raise_for_status()
    chunks, checkpoints = await database_counts(state["document_id"], state["thread_id"])
    checks = {
        "document_after_restart": any(
            item["id"] == state["document_id"] for item in documents.json()
        ),
        "history_after_restart": len(history.json()) >= 6,
        "embeddings_after_restart": chunks > 0,
        "checkpoints_after_restart": checkpoints > 0,
    }
    if not all(checks.values()):
        raise SystemExit(f"Restart verification failed: {json.dumps(checks, sort_keys=True)}")
    print(json.dumps(checks, sort_keys=True))


def main() -> None:
    """Parse the acceptance phase and execute it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-restart", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(verify_restart() if arguments.verify_restart else run_acceptance())


if __name__ == "__main__":
    main()
