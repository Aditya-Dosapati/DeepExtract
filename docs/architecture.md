# Architecture

## System boundaries

The application has four local services and one optional external integration:

| Component | Responsibility | Trust boundary |
| --- | --- | --- |
| `frontend` | Streamlit operations, knowledge-base management, and grounded chat | Uses only the authenticated HTTP API |
| `backend` | FastAPI authentication, ingestion, retrieval, metrics, and LangGraph execution | Owns authorization and business rules |
| `mcp` | Streamable HTTP tools for approved application operations | Verifies the same application JWTs as the API |
| `postgres` | Application records, pgvector embeddings, and graph checkpoints | Private application data store |
| Google Drive MCP | Read-only discovery and download for one configured Drive folder | External Google OAuth boundary |

The local MCP server and Google Drive MCP have different jobs. The local server makes this
application's safe operations available to MCP clients. The remote Drive server is an
ingestion adapter used only by the backend. It is not available to the LangGraph answering
agent and does not receive application JWTs.

```text
Browser ──► Streamlit ──► FastAPI ─────────────────────┐
                                                       ├──► PostgreSQL + pgvector
MCP client ────────────► local MCP server ─────────────┘

selected Drive folder ─► Google Drive MCP ─► sync service ─► ingestion service
```

Application records, vectors, messages, and LangGraph checkpoints are managed through Alembic
migrations. API and MCP processes can restart without losing conversation state.

## Ingestion paths

Manual upload and Google Drive synchronization converge on the same validated ingestion
service.

```text
authenticated upload                  configured Drive folder
        │                                      │
        │                             read-only MCP discovery
        │                                      │
        └──────────► validated bytes ◄─────────┘
                              │
            extension, MIME, signature, and size checks
                              │
                 PDF, DOCX, or UTF-8 extraction
                              │
               normalization and content hashing
                              │
                    page-aware chunking
                              │
                 batched embedding generation
                              │
             documents + chunks + ingestion job
```

The Drive connector traverses only the configured root and optional descendants. It records a
stable manifest entry for each external file. The external file ID prevents duplicate source
items, the modified timestamp skips unchanged versions, and the normalized content hash reuses
an existing owner-scoped document when two files contain the same content. Changed content is
indexed before an orphaned Drive-only document is removed.

Supported Drive inputs are PDF, DOCX, UTF-8 text, and native Google Docs exported as text.
Unsupported formats and oversized files are recorded as skipped. A failure in one file is
recorded without abandoning the remainder of the folder.

If parsing or embedding fails, the document or source item is committed with a sanitized
failure state. Dashboard counts therefore remain useful without exposing provider internals.

## Retrieval and answering

The query is embedded with the configured provider. PostgreSQL applies the owner and optional
thread or metadata filters before ordering chunks by cosine distance. Result count, similarity
threshold, excerpt length, and total context size are bounded by validated configuration.

```text
question
  → decide whether retrieval is required
  → owner- and thread-scoped pgvector search
  → grade the retrieved evidence
  → rewrite and retry at most the configured number of times
  → generate an evidence-bound answer
  → build citations from authorized database records
  → persist user and assistant messages
```

The HNSW index uses `vector_cosine_ops`; vectors are ranked by PostgreSQL rather than loaded
into application memory. The graph uses an owner-and-thread namespace in its PostgreSQL
checkpointer. Provider tokens become LangGraph custom events, FastAPI server-sent events, and
incremental Streamlit output. Streaming and non-streaming endpoints use the same evidence and
persistence boundaries.

Retrieved text is untrusted evidence. The prompt tells the model not to follow instructions in
documents, while the backend independently constructs citations from authorized chunk records.

## Authentication and data isolation

The API issues short-lived JWT access tokens after password authentication. The local MCP
server verifies the same signature, issuer metadata, expiration, and active user. Neither API
payloads nor local MCP tool schemas accept an owner identifier.

Owner filters are applied inside document, thread, message, source-manifest, and vector queries.
Foreign resources return the same not-found behavior as missing resources. Google OAuth tokens
remain backend configuration and are never passed to the browser, local MCP clients, LangGraph,
logs, or stored document metadata.

## Operational visibility

The authenticated metrics endpoint aggregates document, chunk, byte, ingestion, conversation,
and message counts by owner. Public liveness and readiness probes separately report process,
database, and local MCP availability. The dashboard uses these endpoints and never connects
directly to PostgreSQL.

Drive sync returns counts for discovered, indexed, unchanged, duplicate, skipped, and failed
files. These counts describe one synchronous run; durable per-file state is stored in
`source_items`.

## Data model

- `users`: authenticated accounts
- `conversation_threads`: owner-scoped conversation containers
- `messages`: role-based thread history
- `documents`: canonical file metadata and normalized content hashes
- `document_chunks`: source text, hashes, metadata, and embeddings
- `ingestion_jobs`: observable upload and indexing outcomes
- `source_items`: owner-scoped external file identity, version, status, and document mapping
- `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`: durable LangGraph state

Foreign keys cascade private dependent data. A Drive manifest uses `SET NULL` for its document
reference so synchronization can safely repair or re-index a removed canonical document.

## Deployment posture

Development runs all four local services with Docker Compose. A production deployment should
use managed PostgreSQL, TLS at the ingress, a secrets manager, private database networking,
managed OAuth refresh, and a single controlled migration step before application rollout.
Backend and frontend images run as an unprivileged user, and local secrets, credential JSON,
raw evaluation state, caches, and Git metadata are excluded from the image build context.
