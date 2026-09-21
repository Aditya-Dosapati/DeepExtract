"""DeepExtract — Personalized RAG Assistant

High-performance, editorial knowledge workspace built with FastAPI,
PostgreSQL/SQLite vector search, Groq, and Sentence Transformers.
"""

import os
from datetime import datetime
from typing import Any

import streamlit as st

from frontend.client import APIClient, APIError

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MCP_HEALTH_URL = os.getenv("MCP_HEALTH_URL", "http://localhost:8001/health")

# -----------------------------------------------------------------------------
# Warm Editorial Design System & Styling
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg-parchment: #F4F0E8;
    --surface-ivory: #FFFDF8;
    --text-charcoal: #252522;
    --text-muted: #77736A;
    --accent-terracotta: #D96C4F;
    --accent-terracotta-hover: #C05B40;
    --accent-olive: #7C8065;
    --highlight-ochre: #E6B85C;
    --border-warm: #D9D1C4;
    --card-shadow: 0 2px 10px rgba(37, 37, 34, 0.04), 0 1px 3px rgba(37, 37, 34, 0.03);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-parchment) !important;
    color: var(--text-charcoal) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #ECE7DC !important;
    border-right: 1px solid var(--border-warm) !important;
}

h1, h2, h3, .editorial-heading {
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 500 !important;
    color: var(--text-charcoal) !important;
    letter-spacing: -0.01em;
}

.stButton>button {
    background-color: var(--surface-ivory) !important;
    color: var(--text-charcoal) !important;
    border: 1px solid var(--border-warm) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease-in-out !important;
}

.stButton>button:hover {
    border-color: var(--accent-terracotta) !important;
    color: var(--accent-terracotta) !important;
    box-shadow: 0 2px 6px rgba(217, 108, 79, 0.12) !important;
}

.stButton>button[kind="primary"], .stButton>button[data-testid="baseButton-primary"] {
    background-color: var(--accent-terracotta) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--accent-terracotta) !important;
    box-shadow: 0 2px 8px rgba(217, 108, 79, 0.25) !important;
}

.stButton>button[kind="primary"]:hover, .stButton>button[data-testid="baseButton-primary"]:hover {
    background-color: var(--accent-terracotta-hover) !important;
    border-color: var(--accent-terracotta-hover) !important;
    color: #FFFFFF !important;
}

.deepextract-card {
    background-color: var(--surface-ivory);
    border: 1px solid var(--border-warm);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    box-shadow: var(--card-shadow);
    margin-bottom: 1rem;
}

.stat-card {
    background-color: var(--surface-ivory);
    border: 1px solid var(--border-warm);
    border-top: 3px solid var(--accent-terracotta);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    box-shadow: var(--card-shadow);
}

.stat-card-olive {
    background-color: var(--surface-ivory);
    border: 1px solid var(--border-warm);
    border-top: 3px solid var(--accent-olive);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    box-shadow: var(--card-shadow);
}

.stat-card-ochre {
    background-color: var(--surface-ivory);
    border: 1px solid var(--border-warm);
    border-top: 3px solid var(--highlight-ochre);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    box-shadow: var(--card-shadow);
}

.stat-value {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2rem;
    font-weight: 600;
    color: var(--text-charcoal);
    line-height: 1.1;
    margin-top: 0.25rem;
}

.stat-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 4px;
    letter-spacing: 0.02em;
}

.badge-terracotta {
    background-color: rgba(217, 108, 79, 0.12);
    color: var(--accent-terracotta);
    border: 1px solid rgba(217, 108, 79, 0.25);
}

.badge-olive {
    background-color: rgba(124, 128, 101, 0.14);
    color: var(--accent-olive);
    border: 1px solid rgba(124, 128, 101, 0.3);
}

.badge-neutral {
    background-color: #EAE5DB;
    color: var(--text-muted);
    border: 1px solid var(--border-warm);
}

.citation-card {
    background-color: #F8F5EE;
    border: 1px solid var(--border-warm);
    border-left: 3px solid var(--accent-terracotta);
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.85rem;
}

.stTextInput>div>div>input, .stSelectbox>div>div, .stTextArea>div>div>textarea {
    background-color: var(--surface-ivory) !important;
    border: 1px solid var(--border-warm) !important;
    border-radius: 6px !important;
    color: var(--text-charcoal) !important;
}

.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
    border-color: var(--accent-terracotta) !important;
    box-shadow: 0 0 0 1px var(--accent-terracotta) !important;
}

::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-parchment);
}
::-webkit-scrollbar-thumb {
    background: var(--border-warm);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}
</style>
"""


def format_bytes(value: int) -> str:
    """Format byte counts cleanly for operators."""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def get_client() -> APIClient:
    """Return an API client configured with current session token."""
    token = st.session_state.get("token")
    return APIClient(BACKEND_URL, str(token) if token else None)


# -----------------------------------------------------------------------------
# Authentication Screen
# -----------------------------------------------------------------------------
def render_authentication() -> None:
    """Render the editorial DeepExtract login & registration workspace."""
    st.markdown(
        """
        <div style="text-align: center; margin-top: 2rem; margin-bottom: 2rem;">
            <div style="font-family: 'Newsreader', serif; font-size: 2.75rem; font-weight: 500;
                        color: #252522; letter-spacing: -0.02em;">
                DeepExtract
            </div>
            <div style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.12em;
                        color: #D96C4F; font-weight: 600; margin-top: 0.25rem;">
                Personalized RAG Assistant
            </div>
            <div style="font-size: 0.95rem; color: #77736A; max-width: 480px;
                        margin: 0.75rem auto 0 auto; line-height: 1.5;">
                Source-grounded document intelligence backed by local Sentence Transformers
                and Groq LLM.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col2, _ = st.columns([1, 1.6, 1])
    with col2:
        with st.container(border=True):
            tab_login, tab_register = st.tabs(["Sign in", "Create account"])
            client = APIClient(BACKEND_URL)

            with tab_login:
                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                with st.form("form_signin"):
                    email = st.text_input(
                        "Work email",
                        placeholder="name@company.com",
                        key="auth_login_email",
                    )
                    password = st.text_input(
                        "Password",
                        type="password",
                        key="auth_login_password",
                    )
                    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                    submitted = st.form_submit_button(
                        "Sign in to workspace",
                        type="primary",
                        use_container_width=True,
                    )
                    if submitted:
                        if not email or not password:
                            st.error("Please provide both email and password.")
                        else:
                            try:
                                with st.spinner("Authenticating..."):
                                    token = client.login(email.strip(), password)
                                st.session_state.token = token
                                st.session_state.cached_user = None
                                st.session_state.cached_docs = None
                                st.session_state.cached_threads = None
                                st.rerun()
                            except APIError as exc:
                                st.error(str(exc))

            with tab_register:
                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                with st.form("form_signup"):
                    username = st.text_input(
                        "Full name or username",
                        placeholder="Jane Doe",
                        key="auth_reg_user",
                    )
                    email = st.text_input(
                        "Email address",
                        placeholder="name@company.com",
                        key="auth_reg_email",
                    )
                    password = st.text_input(
                        "Password (min. 8 characters)",
                        type="password",
                        key="auth_reg_password",
                    )
                    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                    submitted = st.form_submit_button(
                        "Create new account",
                        use_container_width=True,
                    )
                    if submitted:
                        if not username or not email or not password:
                            st.error("All fields are required.")
                        elif len(password) < 8:
                            st.error("Password must be at least 8 characters.")
                        else:
                            try:
                                with st.spinner("Creating account..."):
                                    client.register(username.strip(), email.strip(), password)
                                st.success("Account created successfully. Please sign in.")
                            except APIError as exc:
                                st.error(str(exc))


# -----------------------------------------------------------------------------
# Main Dashboard View
# -----------------------------------------------------------------------------
def render_dashboard(client: APIClient, user: dict[str, Any]) -> None:
    """Render the personalized overview dashboard."""
    username = user.get("username", "Researcher")
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.25rem; margin-bottom: 0.2rem;">Good day, {username}</h1>
            <div style="color: var(--text-muted); font-size: 0.95rem;">
                Your DeepExtract personalized research workspace is active and synchronized.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        metrics = client.metrics()
    except APIError as exc:
        st.error(f"Failed to load metrics: {exc}")
        return

    kb = metrics["knowledge_base"]
    act = metrics["activity"]

    # Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Indexed Documents</div>
                <div class="stat-value">{kb["indexed_documents"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="stat-card-olive">
                <div class="stat-label">Vector Chunks (384-D)</div>
                <div class="stat-value">{kb["indexed_chunks"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="stat-card-ochre">
                <div class="stat-label">Storage Consumed</div>
                <div class="stat-value">{format_bytes(kb["stored_bytes"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Research Threads</div>
                <div class="stat-value">{act["threads"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Quick Action Bar
    st.markdown("### Quick actions")
    qa_col1, qa_col2, qa_col3 = st.columns(3)
    with qa_col1:
        if st.button("💬 Start research chat", use_container_width=True, type="primary"):
            st.session_state.current_page = "chat"
            st.rerun()
    with qa_col2:
        if st.button("📁 Upload new document", use_container_width=True):
            st.session_state.current_page = "knowledge"
            st.rerun()
    with qa_col3:
        if st.button("⚡ Inspect system health", use_container_width=True):
            st.session_state.current_page = "health"
            st.rerun()

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Two column layout: Recent Documents & Recent Conversations
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown("### Recent documents")
        try:
            docs = client.documents()
            st.session_state.cached_docs = docs
        except APIError:
            docs = st.session_state.get("cached_docs", [])

        if not docs:
            st.markdown(
                """
                <div class="deepextract-card" style="text-align: center; padding: 2rem;">
                    <div style="font-size: 1.5rem; color: var(--text-muted);
                                margin-bottom: 0.5rem;">📄</div>
                    <div style="font-weight: 600;">No documents in knowledge base</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">
                        Upload PDF, DOCX, or TXT files to ground your research assistant.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for doc in docs[:5]:
                with st.container(border=True):
                    d1, d2, d3 = st.columns([3, 1.5, 1])
                    metadata = doc.get("metadata_json", {})
                    raw_ext = metadata.get("file_extension", ".pdf")
                    ext = raw_ext.upper().replace(".", "")
                    d1.markdown(
                        f"**{doc['display_name']}** <span class='badge badge-neutral'>{ext}</span>",
                        unsafe_allow_html=True,
                    )
                    status_title = str(doc["status"]).title()
                    d2.markdown(
                        f"<span class='badge badge-olive'>{status_title}</span>",
                        unsafe_allow_html=True,
                    )
                    d3.caption(format_bytes(int(doc["file_size"])))

    with right_col:
        st.markdown("### Active conversations")
        try:
            threads = client.threads()
            st.session_state.cached_threads = threads
        except APIError:
            threads = st.session_state.get("cached_threads", [])

        if not threads:
            st.markdown(
                """
                <div class="deepextract-card" style="text-align: center; padding: 2rem;">
                    <div style="font-size: 1.5rem; color: var(--text-muted);
                                margin-bottom: 0.5rem;">💬</div>
                    <div style="font-weight: 600;">No active conversations</div>
                    <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem;">
                        Create a thread to begin interactive Q&A.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for thread in threads[:5]:
                with st.container(border=True):
                    t1, t2 = st.columns([3, 1])
                    t1.markdown(f"**{thread['title']}**")
                    if t2.button(
                        "Open",
                        key=f"open_dash_th_{thread['id']}",
                        use_container_width=True,
                    ):
                        st.session_state.active_thread_id = str(thread["id"])
                        st.session_state.current_page = "chat"
                        st.rerun()


# -----------------------------------------------------------------------------
# Research Chat Interface
# -----------------------------------------------------------------------------
def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render formatted source citations with relevance scores and excerpts."""
    if not sources:
        return
    st.markdown(
        f"<div style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; "
        f"color: var(--text-muted); margin-top: 0.75rem;'>Grounded Sources ({len(sources)})</div>",
        unsafe_allow_html=True,
    )
    for idx, source in enumerate(sources, 1):
        page = f", Page {source['page_number']}" if source.get("page_number") else ""
        score_pct = int(source["score"] * 100)
        with st.expander(f"Source #{idx}: {source['document_name']}{page} · {score_pct}% match"):
            st.markdown(
                f"<div class='citation-card'>{source['excerpt']}</div>",
                unsafe_allow_html=True,
            )


def render_chat_interface(client: APIClient, user: dict[str, Any]) -> None:
    """Render the grounded research chat interface with streaming and citations."""
    del user  # Unused in chat interface

    try:
        threads = client.threads()
        st.session_state.cached_threads = threads
    except APIError as exc:
        st.error(f"Error fetching threads: {exc}")
        threads = st.session_state.get("cached_threads", [])

    header_col, action_col = st.columns([3, 1.5])
    with header_col:
        st.markdown(
            """
            <h1 style="font-size: 2rem; margin-bottom: 0.2rem;">Research Chat</h1>
            <div style="color: var(--text-muted); font-size: 0.9rem;">
                Ask questions grounded strictly in your indexed documents.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with action_col:
        with st.popover("+ New thread", use_container_width=True):
            new_title = st.text_input("Thread title", value="Document Research Consultation")
            if st.button("Create conversation", type="primary", use_container_width=True):
                try:
                    created = client.create_thread(
                        new_title.strip() or "Document Research Consultation"
                    )
                    st.session_state.active_thread_id = str(created["id"])
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))

    if not threads:
        st.markdown(
            """
            <div class="deepextract-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.5rem;
                            font-weight: 500;">No conversations yet</div>
                <div style="color: var(--text-muted); max-width: 450px;
                            margin: 0.5rem auto 1.5rem auto; font-size: 0.9rem;">
                    Create a conversation thread to query your indexed business documents
                    with verifiable source citations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Create your first conversation", type="primary"):
            try:
                created = client.create_thread("Initial Document Consultation")
                st.session_state.active_thread_id = str(created["id"])
                st.rerun()
            except APIError as exc:
                st.error(str(exc))
        return

    # Select Active Thread
    thread_map = {str(t["id"]): f"{t['title']} ({str(t['id'])[:8]})" for t in threads}
    active_thread_id = st.session_state.get("active_thread_id")
    if not active_thread_id or active_thread_id not in thread_map:
        active_thread_id = str(threads[0]["id"])
        st.session_state.active_thread_id = active_thread_id

    th_col1, th_col2 = st.columns([4, 1])
    with th_col1:
        active_thread_id = st.selectbox(
            "Active Conversation",
            options=list(thread_map.keys()),
            format_func=lambda tid: thread_map[tid],
            key="chat_thread_selector",
            label_visibility="collapsed",
        )
        st.session_state.active_thread_id = active_thread_id
    with th_col2:
        with st.popover("Delete", use_container_width=True, help="Delete this conversation thread"):
            st.markdown("**Delete conversation?**")
            st.caption("Permanently delete this chat thread and its entire message history.")
            if st.button(
                "Confirm Delete",
                key="confirm_del_thread",
                type="primary",
                use_container_width=True,
            ):
                try:
                    client.delete_thread(active_thread_id)
                    st.session_state.active_thread_id = None
                    st.session_state.cached_threads = None
                    st.toast("Conversation deleted", icon="🗑️")
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))

    st.markdown(
        "<hr style='margin: 1rem 0; border: none; border-top: 1px solid var(--border-warm);'>",
        unsafe_allow_html=True,
    )

    # Load and render message history
    try:
        history = client.history(active_thread_id)
    except APIError as exc:
        st.error(f"Failed to load thread history: {exc}")
        history = []

    if not history:
        st.markdown(
            """
            <div style="text-align: center; padding: 1.5rem 1rem;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.35rem; color: #252522;">
                    What would you like to explore?
                </div>
                <div style="color: #77736A; font-size: 0.9rem; margin-top: 0.25rem;">
                    Select a starter question or enter your own query below.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; "
            "color: var(--text-muted); margin-bottom: 0.5rem;'>Suggested Inquiries</div>",
            unsafe_allow_html=True,
        )
        sq1, sq2 = st.columns(2)
        with sq1:
            if st.button(
                "What embedding model and dimension does the platform use?",
                use_container_width=True,
            ):
                st.session_state.pending_query = (
                    "What embedding model and dimension does the platform use?"
                )
                st.rerun()
            if st.button(
                "How is user authentication and data isolation handled?",
                use_container_width=True,
            ):
                st.session_state.pending_query = (
                    "How is user authentication and data isolation handled?"
                )
                st.rerun()
        with sq2:
            if st.button(
                "What is the retrieval SLA and system availability target?",
                use_container_width=True,
            ):
                st.session_state.pending_query = (
                    "What is the retrieval SLA and system availability target?"
                )
                st.rerun()
            if st.button(
                "Summarize the key policies and architecture constraints",
                use_container_width=True,
            ):
                st.session_state.pending_query = (
                    "Summarize the key policies and architecture constraints"
                )
                st.rerun()

    # Render History Messages
    for msg in history:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        sources = msg.get("sources", [])

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)
                if sources:
                    render_sources(sources)

    # Process new prompt or pending suggestion
    query_to_send = st.chat_input("Ask about your indexed documents...")
    if not query_to_send and st.session_state.get("pending_query"):
        query_to_send = st.session_state.pop("pending_query")

    if query_to_send:
        with st.chat_message("user"):
            st.markdown(query_to_send)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Searching knowledge base & synthesizing verified evidence..."):
                answer = ""
                completed_payload = None
                try:
                    for event, payload in client.stream_chat(active_thread_id, query_to_send):
                        if event == "token":
                            answer += str(payload)
                            placeholder.markdown(f"{answer}▌")
                        elif event == "complete" and isinstance(payload, dict):
                            completed_payload = payload
                        elif event == "error" and isinstance(payload, dict):
                            raise APIError(str(payload.get("message", "Stream failed")))
                    placeholder.markdown(answer)
                    if completed_payload and completed_payload.get("sources"):
                        render_sources(completed_payload["sources"])
                except APIError as exc:
                    placeholder.error(f"Error: {exc}")


# -----------------------------------------------------------------------------
# Knowledge Base (Document Management)
# -----------------------------------------------------------------------------
def render_knowledge_base(client: APIClient, user: dict[str, Any]) -> None:
    """Render document upload, ingestion state, and collection management."""
    del user  # Unused in knowledge base

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2rem; margin-bottom: 0.2rem;">Knowledge Base</h1>
            <div style="color: var(--text-muted); font-size: 0.9rem;">
                Upload, index, and organize documents with local Sentence Transformers.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Document Upload Card
    with st.container(border=True):
        st.markdown("### Upload and index documents")
        st.caption(
            "Supported formats: **PDF, DOCX, TXT** (Max 25 MB per file). "
            "Clean text is chunked and embedded with 384 dimensions."
        )

        try:
            threads = client.threads()
        except APIError:
            threads = []

        thread_options = {"Global (All conversations)": ""} | {
            f"{t['title']} ({str(t['id'])[:8]})": str(t["id"]) for t in threads
        }

        with st.form("form_doc_upload", clear_on_submit=True):
            f1, f2 = st.columns([3, 1.5])
            with f1:
                uploaded_file = st.file_uploader(
                    "Select document",
                    type=["pdf", "docx", "txt"],
                    label_visibility="collapsed",
                )
            with f2:
                selected_scope = st.selectbox(
                    "Conversation scope",
                    options=list(thread_options.keys()),
                )

            submit_upload = st.form_submit_button(
                "Index document",
                type="primary",
                use_container_width=True,
            )
            if submit_upload and uploaded_file is not None:
                try:
                    with st.spinner(f"Parsing and embedding {uploaded_file.name}..."):
                        result = client.upload(
                            file_name=uploaded_file.name,
                            data=uploaded_file.getvalue(),
                            mime_type=uploaded_file.type or "application/octet-stream",
                            thread_id=thread_options[selected_scope],
                        )
                    if result.get("duplicate"):
                        st.info("Document already exists in your knowledge base.")
                    else:
                        st.success(
                            f"Successfully indexed {uploaded_file.name} "
                            f"({result.get('chunks_created', 0)} chunks)."
                        )
                    st.session_state.cached_docs = None
                    st.rerun()
                except APIError as exc:
                    st.error(f"Upload failed: {exc}")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Document Library with Search & Filter
    st.markdown("### Document library")

    try:
        documents = client.documents()
        st.session_state.cached_docs = documents
    except APIError as exc:
        st.error(f"Failed to fetch documents: {exc}")
        documents = st.session_state.get("cached_docs", [])

    if not documents:
        st.markdown(
            """
            <div class="deepextract-card" style="text-align: center; padding: 2.5rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📁</div>
                <div style="font-family: 'Newsreader', serif; font-size: 1.25rem;">
                    Knowledge base is empty
                </div>
                <div style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.25rem;">
                    Upload files above to begin grounding your assistant.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Real-time search filter
    search_term = st.text_input(
        "Search indexed documents",
        placeholder="Filter by document name or format...",
        label_visibility="collapsed",
    )

    filtered_docs = [
        d for d in documents if not search_term or search_term.lower() in d["display_name"].lower()
    ]

    st.caption(f"Showing {len(filtered_docs)} of {len(documents)} indexed documents")

    for doc in filtered_docs:
        with st.container(border=True):
            col_name, col_status, col_size, col_date, col_action = st.columns(
                [3.5, 1.5, 1.2, 1.8, 1]
            )
            metadata = doc.get("metadata_json", {})
            raw_ext = metadata.get("file_extension", ".pdf")
            ext = raw_ext.upper().replace(".", "")
            col_name.markdown(
                f"**{doc['display_name']}** <span class='badge badge-neutral'>{ext}</span>",
                unsafe_allow_html=True,
            )
            status_style = "badge-olive" if doc["status"] == "completed" else "badge-terracotta"
            status_text = str(doc["status"]).title()
            col_status.markdown(
                f"<span class='badge {status_style}'>{status_text}</span>",
                unsafe_allow_html=True,
            )
            col_size.write(format_bytes(int(doc["file_size"])))

            created_at = doc.get("created_at")
            if created_at:
                try:
                    formatted_dt = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    ).strftime("%b %d, %H:%M")
                    col_date.caption(f"Added {formatted_dt}")
                except Exception:
                    col_date.caption("Indexed")

            with col_action:
                with st.popover("Delete", use_container_width=True):
                    st.markdown("**Confirm deletion?**")
                    st.caption(
                        f"Permanently remove **{doc['display_name']}**, all vector embeddings, "
                        "and associated chunk records from the knowledge base."
                    )
                    if st.button(
                        "Confirm Delete",
                        key=f"confirm_del_doc_{doc['id']}",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            client.delete_document(str(doc["id"]))
                            st.session_state.cached_docs = None
                            st.toast(f"Deleted {doc['display_name']}", icon="🗑️")
                            st.rerun()
                        except APIError as exc:
                            st.error(f"Failed to delete document: {exc}")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Google Drive Folder Sync
    with st.expander("Google Drive Folder Synchronization"):
        st.markdown(
            "Synchronize business documents directly from your server-configured "
            "Google Drive folder. Unchanged files are skipped automatically."
        )
        if st.button("Trigger Drive synchronization", type="primary"):
            try:
                with st.spinner("Syncing Google Drive documents..."):
                    summary = client.sync_google_drive()
                st.success(
                    f"Sync complete: {summary.get('discovered', 0)} discovered, "
                    f"{summary.get('indexed', 0)} indexed."
                )
                st.session_state.cached_docs = None
                st.rerun()
            except APIError as exc:
                st.error(f"Google Drive sync failed: {exc}")


# -----------------------------------------------------------------------------
# System Health & Infrastructure
# -----------------------------------------------------------------------------
def render_system_health(client: APIClient, user: dict[str, Any]) -> None:
    """Render live probes, latency diagnostics, and model configuration."""
    del user  # Unused in system health

    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2rem; margin-bottom: 0.2rem;">System Health & Infrastructure</h1>
            <div style="color: var(--text-muted); font-size: 0.9rem;">
                Real-time service probes, vector engine status, and provider telemetry.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Refresh probes", type="primary"):
        st.rerun()

    live, live_ms = client.health("/health/live")
    ready, ready_ms = client.health("/health/ready")
    mcp_ready, mcp_ms = client.service_health(MCP_HEALTH_URL)

    c1, c2, c3 = st.columns(3)
    with c1:
        color_api = "#7C8065" if live else "#D96C4F"
        status_api = "Healthy" if live else "Offline"
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">FastAPI Backend API</div>
                <div class="stat-value" style="color: {color_api};">{status_api}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem;">
                    Latency: {live_ms:.1f} ms
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        color_db = "#7C8065" if ready else "#D96C4F"
        status_db = "Ready" if ready else "Offline"
        st.markdown(
            f"""
            <div class="stat-card-olive">
                <div class="stat-label">Vector Database Store</div>
                <div class="stat-value" style="color: {color_db};">{status_db}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem;">
                    Latency: {ready_ms:.1f} ms · 384 dimensions
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        color_mcp = "#7C8065" if mcp_ready else "#77736A"
        status_mcp = "Active" if mcp_ready else "Standby"
        st.markdown(
            f"""
            <div class="stat-card-ochre">
                <div class="stat-label">MCP Protocol Server</div>
                <div class="stat-value" style="color: {color_mcp};">{status_mcp}</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem;">
                    Latency: {mcp_ms:.1f} ms
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Active Model Architecture")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Embedding Model Provider**")
            st.markdown("`Sentence Transformers (BAAI/bge-small-en-v1.5)`")
            st.caption("384-dimensional dense vector embeddings executed locally offline.")
        with m2:
            st.markdown("**Language Model (LLM)**")
            st.markdown("`Groq (qwen/qwen3.8-27b)`")
            st.caption("Source-grounded prompt structured with JSON untrusted evidence guards.")


# -----------------------------------------------------------------------------
# Main Application Flow & Sidebar
# -----------------------------------------------------------------------------
def main() -> None:
    """Main application lifecycle and layout router."""
    st.set_page_config(
        page_title="DeepExtract — Personalized RAG Assistant",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Check authentication
    if not st.session_state.get("token"):
        render_authentication()
        return

    client = get_client()

    # Load or cache current user
    if not st.session_state.get("cached_user"):
        try:
            st.session_state.cached_user = client.current_user()
        except APIError:
            st.session_state.clear()
            st.rerun()
            return

    user = st.session_state.cached_user

    # Navigation State
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"

    # Render Editorial Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0;">
                <div style="font-family: 'Newsreader', serif; font-size: 1.75rem;
                            font-weight: 600; color: #252522; letter-spacing: -0.01em;">
                    DeepExtract
                </div>
                <div style="font-size: 0.75rem; text-transform: uppercase;
                            letter-spacing: 0.1em; color: #D96C4F; font-weight: 600;">
                    Personalized RAG Assistant
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user_name = user.get("username", "User")
        user_email = user.get("email", "")
        st.markdown(
            f"""
            <div style="background-color: #FFFDF8; border: 1px solid #D9D1C4;
                        border-radius: 6px; padding: 0.75rem; margin-bottom: 1.25rem;">
                <div style="font-weight: 600; font-size: 0.9rem; color: #252522;">{user_name}</div>
                <div style="font-size: 0.75rem; color: #77736A;">{user_email}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size: 0.75rem; font-weight: 600; text-transform: uppercase; "
            "color: #77736A; margin-bottom: 0.5rem;'>Navigation</div>",
            unsafe_allow_html=True,
        )

        pages = {
            "dashboard": "📊 Overview Dashboard",
            "chat": "💬 Research Chat",
            "knowledge": "📁 Knowledge Base",
            "health": "⚡ System Status",
        }

        for page_key, page_label in pages.items():
            is_active = st.session_state.current_page == page_key
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                page_label,
                key=f"nav_{page_key}",
                use_container_width=True,
                type=btn_type,
            ):
                st.session_state.current_page = page_key
                st.rerun()

        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<hr style='border: none; border-top: 1px solid #D9D1C4; margin: 1rem 0;'>",
            unsafe_allow_html=True,
        )

        if st.button("🚪 Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Route to selected page
    page = st.session_state.current_page
    if page == "dashboard":
        render_dashboard(client, user)
    elif page == "chat":
        render_chat_interface(client, user)
    elif page == "knowledge":
        render_knowledge_base(client, user)
    elif page == "health":
        render_system_health(client, user)


if __name__ == "__main__":
    main()
