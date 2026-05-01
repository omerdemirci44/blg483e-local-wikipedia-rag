# Production Recommendation: Local Wikipedia RAG Assistant

## Executive Summary

The current system is a simplified local RAG assistant built for academic
demonstration. It successfully shows the core RAG workflow: Wikipedia ingestion,
manual cleanup and chunking, local embeddings, Chroma retrieval, local Ollama
generation, and a Streamlit UI. It should not be considered production-ready,
but it is a solid MVP for learning and evaluation.

## Current MVP Architecture

- `requests` and BeautifulSoup fetch selected Wikipedia pages.
- Raw article JSON is stored under `data/raw/`.
- Cleaned chunks are stored in `data/processed/chunks.jsonl`.
- Ollama `nomic-embed-text` generates local embeddings.
- Chroma stores vectors and metadata under `chroma_db/`.
- Rule-based classification selects person, place, mixed, or broad retrieval.
- Ollama `llama3.2:3b` generates grounded answers.
- Streamlit provides a local chat interface.

## Production Deployment Recommendation

For a production system, separate the ingestion pipeline, vector database, model
runtime, API service, and UI into independently managed services. Use controlled
deployment environments, explicit model versioning, observability, backups, and
security review before exposing the system to real users.

The current project should remain local for the homework. It does not include
authentication, authorization, rate limiting, monitoring, or hardened data
management.

## Data Ingestion Improvements

- Use Wikipedia APIs or dumps for more stable ingestion.
- Store article revision IDs and retrieval timestamps.
- Add retry policies and structured ingestion logs.
- Validate article quality and handle redirects explicitly.
- Add a configurable topic list instead of hardcoded entities.
- Support incremental refresh instead of full overwrite.

## Chunking Improvements

- Evaluate paragraph-aware and section-aware chunking.
- Preserve section headings in chunk metadata.
- Avoid splitting tables or lists without structure.
- Track original character offsets for source highlighting.
- Tune chunk size and overlap based on retrieval evaluation.

## Embedding And Vector Store Improvements

- Record embedding model name and version in metadata.
- Rebuild only chunks whose text changed.
- Add vector-store backup and restore procedures.
- Compare Chroma settings for persistence and retrieval performance.
- Evaluate alternative local embedding models for quality and latency.

## Retrieval Ranking Improvements

- Add hybrid search with keyword matching plus vector search.
- Add reranking for top retrieved chunks.
- Improve handling of comparative questions with balanced entity coverage.
- Add filters for title, type, source, and section.
- Measure recall on a small evaluation question set.

## Answer Quality Improvements

- Add citation markers in generated answers.
- Highlight which source chunks support each sentence.
- Add stronger answer validation before showing final output.
- Experiment with prompt variants and local model choices.
- Use structured output for answer, confidence, and cited chunks.

## Security And Privacy Considerations

- Keep user questions and generated outputs local unless explicitly exported.
- Avoid sending private user data to external services.
- Sanitize logs if user questions may contain personal information.
- Validate any future file upload feature before indexing content.
- Do not expose the local Streamlit app publicly without authentication and
  network hardening.

## Reliability And Monitoring

- Add structured logs for ingestion, chunking, embedding, retrieval, and
  generation.
- Track Ollama availability and model load failures.
- Record latency for embedding, retrieval, and generation.
- Add health checks for Chroma collection readiness.
- Add regression tests for known supported and unsupported questions.

## Scalability Considerations

- The current dataset is intentionally small.
- Larger datasets would require batching, incremental indexing, and stronger
  storage management.
- Embedding generation should be queued or parallelized carefully.
- Retrieval latency should be measured as collection size grows.
- The Streamlit UI is appropriate for demos, not high-concurrency use.

## Known Limitations

- The dataset is fixed to selected famous people and places.
- Classification is rule-based and may miss ambiguous queries.
- Retrieval quality depends on local embedding behavior.
- Generation quality depends on the installed Ollama model.
- The system can still retrieve weak context for unknown questions, though the
  generation guard returns `I don't know.` for clear unsupported cases.
- The project has no production authentication, monitoring, or deployment
  automation.

## Possible Future Extensions

- Streaming responses in the Streamlit UI.
- Citations and source highlighting inside answers.
- Chat memory for multi-turn questions.
- Model comparison across local LLMs and embedding models.
- Latency measurement and simple performance dashboards.
- Query and embedding caching.
- Better reranking for retrieved chunks.
- Expanded datasets beyond the initial people and places list.
- Evaluation reports with fixed benchmark questions.
