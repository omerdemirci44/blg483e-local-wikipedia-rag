"""Embedding and Chroma storage placeholders.

The final implementation will generate embeddings locally and persist them in
Chroma. Sprint 0 keeps this module import-safe by avoiding Chroma/Ollama imports.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def get_vector_store_path() -> Path:
    """Return the planned local directory for Chroma persistence."""

    return config.CHROMA_DB_DIR


def create_or_load_collection(collection_name: str = config.COLLECTION_NAME) -> None:
    """Placeholder for creating or loading a Chroma collection."""

    # TODO: Sprint 3 - initialize Chroma persistent client and collection.
    _ = collection_name
    return None


def embed_chunks(chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    """Placeholder for attaching local embeddings to chunk records."""

    # TODO: Sprint 3 - call the local embedding model through Ollama.
    _ = chunks
    return []


def store_embeddings(chunks_with_embeddings: list[dict[str, object]]) -> int:
    """Placeholder for storing embedded chunks in Chroma."""

    # TODO: Sprint 3 - upsert documents, embeddings, ids, and metadata.
    _ = chunks_with_embeddings
    return 0
