"""Retrieval placeholders for the local RAG assistant."""

from __future__ import annotations

from . import config


def retrieve_context(
    query: str,
    category: str | None = None,
    top_k: int = config.DEFAULT_TOP_K,
) -> list[dict[str, object]]:
    """Placeholder for retrieving relevant chunks from Chroma."""

    # TODO: Sprint 4 - query Chroma using local embeddings and metadata filters.
    _ = (query, category, top_k)
    return []


def format_sources(context_chunks: list[dict[str, object]]) -> list[str]:
    """Return source labels for retrieved context chunks."""

    # TODO: Sprint 4 - format article titles and chunk identifiers.
    _ = context_chunks
    return []
