# Delivery roadmap

## Current release boundary

The current system provides authenticated manual uploads, a folder-scoped Google Drive MCP
sync, durable pgvector indexing, LangGraph retrieval and answering, citations, dashboard
metrics, health checks, and local MCP tools. The Drive integration is manually triggered and
uses a short-lived access token. These are deliberate release boundaries, not hidden automated
behaviors.

## Next milestone: validate 1,000 business documents

The next release should prove that the pipeline remains correct and recoverable with 1,000
representative documents before adding more connectors.

1. Build a permission-approved corpus containing PDF, DOCX, TXT, and native Google Docs, with
   known duplicates, changed versions, unsupported files, and controlled parser failures.
2. Record a manifest of expected file IDs, versions, normalized hashes, document mappings, and
   supported or skipped outcomes.
3. Run a clean initial sync, an unchanged repeat sync, a partial-failure retry, and an update
   sync. Verify that no file is silently lost and no unchanged content is embedded twice.
4. Measure discovery, download, parsing, embedding, database-write, and end-to-end duration;
   record provider rate-limit behavior and peak memory rather than choosing an SLO without data.
5. Create domain-expert questions with exact gold evidence across the corpus. Measure retrieval,
   answer grounding, refusals, prompt-injection resistance, and latency on an isolated account.
6. Load-test concurrent retrieval and chat while a sync is active. Confirm tenant isolation,
   citation validity, checkpoint durability, and predictable recovery after process restart.

The milestone is complete when all 1,000 source records have an expected terminal state,
repeated syncs are idempotent, failures can resume without manual database repair, every
security-isolation test passes, and the measured quality/latency baseline is committed without
credentials or private document text.

## Production work after the scale test

- Move ingestion to a queue-backed worker with retry budgets, progress, and cancellation.
- Replace manual Drive sync with managed OAuth refresh and scheduled or change-token-based sync.
- Add OCR, spreadsheet/slides parsing, and table-aware document structure where required by the
  measured corpus.
- Evaluate hybrid lexical/vector retrieval, reranking, and adaptive context selection.
- Deploy to GCP with private Cloud SQL connectivity, Secret Manager, TLS ingress, backups, and a
  single migration job.
- Add OpenTelemetry traces and provider cost, rate-limit, and latency monitoring.
- Define retention, deletion, audit export, and connector revocation procedures.
