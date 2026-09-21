# DeepExtract — Personalized RAG Assistant

> **A high-performance, source-grounded knowledge assistant powered by FastAPI, LangGraph, local Sentence Transformers embeddings, Groq LLM, and PostgreSQL + pgvector (with SQLite fallback).**

DeepExtract transforms your enterprise documents (PDF, DOCX, TXT) into an interactive, grounded research workspace. Every answer is synthesized strictly from validated knowledge chunks with verifiable citations, eliminating hallucinations and ensuring complete data isolation.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features & Core Capabilities](#-key-features--core-capabilities)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Project Folder Structure](#-project-folder-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Environment Configuration](#-environment-configuration)
- [Running the Application Locally](#-running-the-application-locally)
- [Database Configuration & Fallback Engine](#-database-configuration--fallback-engine)
- [API Reference](#-api-reference)
- [Model Context Protocol (MCP) Integration](#-model-context-protocol-mcp-integration)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Security & Tenant Isolation](#-security--tenant-isolation)
- [Performance & Optimization](#-performance--optimization)
- [Known Limitations & Roadmap](#-known-limitations--roadmap)
- [License](#-license)

---

## 🔍 Overview

**DeepExtract** is an enterprise-grade Agentic Retrieval-Augmented Generation (RAG) platform. It allows users to upload documents, automatically extract clean text, compute dense vector representations locally on CPU/GPU without external API overhead, and chat with an AI assistant that cites exact source pages and chunks.

The platform provides:
1. **Interactive Streamlit Workspace**: A warm, editorial UI with dedicated views for Overview Dashboard, Research Chat, Knowledge Base, and System Health.
2. **RESTful API**: Fast, async OpenAPI endpoints for document ingestion, retrieval, conversation management, and telemetry.
3. **MCP Server**: Standardized Model Context Protocol interface exposing search and grounded answering to external AI agent ecosystems.

---

## ✨ Key Features & Core Capabilities

- **Document Upload & Parsing**: Supports PDF (multi-page text extraction), DOCX (paragraphs and tables), and TXT files up to 25MB per upload.
- **Deterministic Chunking & Content Deduplication**: Normalizes text, splits into bounded chunks (1000 characters with 150-character overlap), and computes SHA-256 content hashes to prevent duplicate chunk storage.
- **Local Dense Vector Embeddings**: Embeds chunks using **`BAAI/bge-small-en-v1.5`** (384 dimensions) locally via Hugging Face Sentence Transformers with L2 normalization—zero external embedding API required.
- **Hybrid Vector Storage & Search**: Native HNSW cosine similarity index in PostgreSQL with pgvector, plus an automated SQLite cosine fallback for isolated local development.
- **LangGraph Agentic Answering**: Bounded LangGraph execution graph that classifies queries, performs semantic retrieval, grades retrieved context, applies bounded query rewriting on low relevance, and generates answers via **Groq** (`llama-3.3-70b-versatile` / `qwen/qwen3.8-27b`) or OpenAI.
- **Verifiable Source Citations**: Assistant answers attach structured evidence chips containing document filename, page number, similarity match percentage, and verbatim chunk excerpts.
- **Document Management & Secure Deletion**: Full document deletion with a confirmation step to prevent accidental loss. Purges document metadata, extracted chunks, vector embeddings, and ingestion jobs across all database engines.
- **Out-of-Domain Query Refusal**: Refuses ungrounded or out-of-context inquiries cleanly without hallucination:
  > *"I could not find enough evidence in your indexed knowledge base to answer that. Try rephrasing the question or upload a relevant document."*
- **Multi-Tenant Security & Isolation**: Argon2 password hashing, JSON Web Tokens (JWT with HS256), and strict `owner_id` scoping across all database queries, checkpoints, and vector searches.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           DeepExtract Frontend                          │
│                  (Streamlit Editorial Research Workspace)               │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP / SSE Stream
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Backend                               │
│       (Auth, Ingestion Pipeline, Search Router, Chat Service)           │
└────────────┬───────────────────────┬──────────────────────┬─────────────┘
             │                       │                      │
             ▼                       ▼                      ▼
┌─────────────────────────┐ ┌──────────────────┐ ┌────────────────────────┐
│ Sentence Transformers   │ │ LangGraph Agent  │ │ MCP Server             │
│ (BAAI/bge-small-en-v1.5)│ │ Workflow Engine  │ │ (Streamable HTTP 8001) │
│ Local 384-D Embeddings  │ └────────┬─────────┘ └────────────────────────┘
└─────────────────────────┘          │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LLM Generation Layer (Groq / OpenAI)               │
│                  llama-3.3-70b-versatile / qwen/qwen3.8-27b             │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Vector & Metadata Storage                         │
│  PostgreSQL 16 + pgvector (HNSW)   ──OR──   Async SQLite (Fallback)     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Component | Technology |
|---|---|---|
| **Frontend** | UI Workspace | Streamlit, Custom Warm Editorial CSS System |
| **API** | REST API & ASGI | FastAPI, Pydantic v2, Pydantic-Settings |
| **Agentic Workflow** | RAG Graph & Orchestration | LangGraph, LangChain Core |
| **Embeddings** | Local Vector Model | Sentence Transformers (`BAAI/bge-small-en-v1.5`, 384-D) |
| **LLM Provider** | Cloud Inference | Groq SDK (`AsyncGroq`), OpenAI SDK |
| **Database** | Metadata & Vectors | PostgreSQL 16 + `pgvector`, SQLAlchemy 2.0 (Async), Alembic |
| **Local Fallback** | Standalone Engine | SQLite + `aiosqlite` with vector cosine similarity |
| **Integration** | Agent Protocol | Model Context Protocol (MCP Python SDK) |
| **Auth & Security** | Access Control | JWT (PyJWT), Argon2 (`pwdlib[argon2]`) |
| **Testing & Quality**| Dev Tooling | pytest, pytest-asyncio, pytest-cov, Ruff, mypy |

---

## 📂 Project Folder Structure

```text
agentic-rag-postgres-mcp-main/
├── .streamlit/
│   └── config.toml           # Streamlit theme and server configuration
├── backend/
│   ├── app/
│   │   ├── agents/           # LangGraph workflow, answer providers, checkpoints
│   │   ├── api/              # FastAPI endpoints (auth, docs, chat, retrieval, health)
│   │   ├── auth/             # JWT token handling and Argon2 password security
│   │   ├── connectors/       # Google Drive MCP connector
│   │   ├── core/             # Application config, logging, errors
│   │   ├── db/               # SQLAlchemy declarative base, async engine, sessions
│   │   ├── ingestion/        # Extractors (PDF/DOCX/TXT), chunking, embeddings
│   │   ├── mcp/              # MCP server, client, tools, and auth
│   │   ├── models/           # SQLAlchemy ORM models (Document, DocumentChunk, User)
│   │   ├── repositories/     # Database access layer (CRUD, documents, threads)
│   │   ├── retrieval/        # Vector search queries (pgvector HNSW & SQLite fallback)
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   └── services/         # Ingestion, chat orchestration, Drive sync
│   ├── migrations/           # Alembic database migrations
│   └── tests/                # Comprehensive unit and integration test suite
├── frontend/
│   ├── app.py                # DeepExtract Streamlit research workspace
│   └── client.py             # Typed API client with connection pooling
├── models/                   # Local cached Sentence Transformer model weights
├── docs/                     # Architecture and roadmap documentation
├── pyproject.toml            # Project dependencies and tool configurations
├── .env.example              # Environment variables template
├── docker-compose.yml        # Multi-container Docker deployment
└── README.md                 # Project documentation
```

---

## 📦 Prerequisites

- **Python 3.12+**
- **uv** (recommended) or **pip** / **venv**
- Optional: **Docker & Docker Compose** (for PostgreSQL + pgvector)

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd agentic-rag-postgres-mcp-main
```

### 2. Create and synchronize the virtual environment
Using `uv`:
```bash
# Synchronize all runtime and frontend dependencies
uv sync --extra frontend
```

Or using standard Python `venv`:
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -e ".[frontend]"
```

---

## ⚙️ Environment Configuration

Copy the example environment template to create `.env`:
```bash
cp .env.example .env
```

Edit `.env` to configure your credentials:

```dotenv
# Application runtime
APP_ENV=development
LOG_LEVEL=INFO

# Database (Leave blank for SQLite local fallback, or provide Postgres URL)
DATABASE_URL=sqlite+aiosqlite:///agentic_rag.db

# Authentication (Generate at least 32 random characters)
JWT_SECRET=your-random-32-character-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Local Embeddings (Sentence Transformers — 384 dimensions)
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384

# LLM (Groq)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_KEY=gsk_your_groq_api_key_here
```

---

## 💻 Running the Application Locally

### Option A: Standalone Local Mode (Fastest Setup)
Runs the complete stack locally using SQLite vector search and local Sentence Transformers.

1. **Start the FastAPI Backend** (Terminal 1):
   ```bash
   # Windows PowerShell
   $env:PYTHONPATH = "."
   .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

   # Linux / macOS
   PYTHONPATH=. uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

2. **Start the DeepExtract Streamlit Workspace** (Terminal 2):
   ```bash
   # Windows PowerShell
   .\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501

   # Linux / macOS
   uv run streamlit run frontend/app.py --server.port 8501
   ```

3. **Open the Workspace**:
   Navigate to **`http://localhost:8501`** in your browser. Create an account or sign in to start uploading documents and querying the knowledge base.

---

### Option B: Docker Compose (PostgreSQL + pgvector)
Runs PostgreSQL 16 with pgvector, FastAPI, MCP Server, and Streamlit in isolated containers:

```bash
# Build and start all services in the background
docker compose up --build -d

# Verify system health
curl http://localhost:8000/health/ready
```

---

## 🗄️ Database Configuration & Fallback Engine

DeepExtract is architected for seamless database portability:

1. **PostgreSQL with pgvector (Production)**:
   - Uses `VECTOR(384)` column type.
   - Computes cosine distance using native HNSW index (`ix_chunks_embedding_hnsw`).
   - Run migrations via Alembic:
     ```bash
     uv run alembic upgrade head
     ```
2. **Async SQLite Engine (Standalone Fallback)**:
   - Activates automatically when `DATABASE_URL=sqlite+aiosqlite:///...`.
   - Stores 384-dimensional vector embeddings in JSON format.
   - Executes exact vector cosine similarity queries in Python, enabling complete offline testing without installing Docker or PostgreSQL.

---

## 📡 API Reference

All REST API endpoints are authenticated under `/api/v1`. Interactive OpenAPI documentation is available at `http://localhost:8000/api/v1/docs`.

| Category | Method & Path | Description |
|---|---|---|
| **Authentication** | `POST /api/v1/auth/register` | Create a new user account |
| **Authentication** | `POST /api/v1/auth/login` | Exchange credentials for JWT access token |
| **User** | `GET /api/v1/users/me` | Retrieve current authenticated user profile |
| **Conversations** | `POST /api/v1/threads` | Create a conversation thread |
| **Conversations** | `GET /api/v1/threads` | List user conversation threads |
| **Conversations** | `DELETE /api/v1/threads/{id}` | Delete a conversation thread and checkpoints |
| **Documents** | `POST /api/v1/documents/upload` | Upload and synchronously index a document |
| **Documents** | `GET /api/v1/documents` | List indexed documents with metadata |
| **Documents** | `DELETE /api/v1/documents/{id}` | Permanently delete document, chunks, and vectors |
| **Retrieval** | `POST /api/v1/retrieval/search` | Execute vector similarity search |
| **Chat** | `POST /api/v1/chat/{thread_id}/stream` | Stream grounded assistant response via SSE |
| **Chat** | `GET /api/v1/chat/{thread_id}/history` | Retrieve conversation message history with sources |
| **Health** | `GET /health/live` · `GET /health/ready` | Process liveness and database readiness probes |

---

## 🔌 Model Context Protocol (MCP) Integration

DeepExtract provides an MCP server implementing the Streamable HTTP transport at `http://localhost:8001/mcp`.

Available MCP Tools:
- **`list_documents`**: List indexed documents for the authenticated user.
- **`search_documents`**: Perform vector semantic search over knowledge chunks.
- **`answer_from_documents`**: Execute the LangGraph grounded Q&A workflow with citations.
- **`ingest_document`**: Index a new document into the knowledge base.

---

## ✅ Testing & Quality Assurance

The codebase maintains a comprehensive suite of unit, integration, and end-to-end tests:

```bash
# Run the complete test suite
uv run pytest

# Run tests with coverage reporting
uv run pytest --cov=backend.app --cov-report=term-missing

# Run code style and lint checks
uv run ruff check .

# Check formatting
uv run ruff format --check .

# Run static type checking
uv run mypy backend/app
```

---

## 🔒 Security & Tenant Isolation

- **Owner-Scoped Authorization**: Every document, chunk, conversation, and vector query is filtered strictly by the authenticated `owner_id`.
- **Untrusted Evidence Isolation**: Context chunks injected into LLM prompts are isolated inside JSON blocks tagged as untrusted data, preventing prompt injection.
- **Argon2 Password Hashing**: Passwords hashed using industry-standard Argon2id with unique salts.
- **Zero Secret Leakage**: No API keys or database passwords are logged, exposed to the frontend, or returned in error payloads.

---

## ⚡ Performance & Optimization

- **Persistent Connection Pooling**: The frontend client uses `httpx.Client` with connection pooling (`max_keepalive_connections=20`), cutting API roundtrip overhead to **<5ms**.
- **Model Pre-Warming**: Sentence Transformers embedding models are pre-warmed during backend lifespan startup, eliminating cold-start latency on the first request.
- **Isolated Sidebar Routing**: Streamlit views execute conditionally, preventing redundant background HTTP requests when typing in chat.
- **Output Token Quota Management**: Configured with `max_tokens=800` to run reliably within LLM rate limit boundaries.

---

## ⚠️ Known Limitations & Roadmap

### Known Limitations
- Scanned PDF documents require pre-processing with OCR (only embedded text is extracted).
- Google Drive sync requires manual trigger with a short-lived OAuth token.

### Roadmap
- [ ] Asynchronous worker queue for massive document ingestion batches (Celery / Redis / ARQ).
- [ ] Hybrid search combining dense vector embeddings with BM25 sparse keyword search and cross-encoder reranking.
- [ ] Managed OAuth 2.0 refresh flow for Google Drive and Microsoft OneDrive connectors.
- [ ] Native OCR support for scanned PDFs and images using Tesseract or cloud OCR engines.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
