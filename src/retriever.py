"""Chroma retrieval for the local Wikipedia RAG assistant."""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from . import config
from .classifier import BOTH_QUERY, PLACE_QUERY, PERSON_QUERY, UNKNOWN_QUERY, classify_query
from .embed_store import OllamaUnavailableError, get_ollama_embedding
from .ingest import PEOPLE_TOPICS, PLACE_TOPICS


VECTOR_STORE_NOT_READY_MESSAGE = (
    "Vector store is not ready. Run python -m src.embed_store first."
)
OLLAMA_NOT_REACHABLE_MESSAGE = (
    "Ollama is not reachable. Make sure Ollama is running and "
    "nomic-embed-text is installed."
)

PLACE_HINTS_BY_KEYWORD = {
    "turkey": ["Hagia Sophia"],
    "türkiye": ["Hagia Sophia"],
    "france": ["Eiffel Tower", "Louvre Museum"],
    "china": ["Great Wall of China"],
    "india": ["Taj Mahal"],
    "italy": ["Colosseum"],
    "jordan": ["Petra"],
    "greece": ["Acropolis of Athens"],
    "japan": ["Mount Fuji"],
    "australia": ["Sydney Opera House"],
}


class VectorStoreNotReadyError(RuntimeError):
    """Raised when the Chroma collection is not available."""


def get_chroma_collection(collection_name: str = config.COLLECTION_NAME) -> Any:
    """Open the existing Chroma collection."""

    if not config.CHROMA_DB_DIR.exists():
        raise VectorStoreNotReadyError(VECTOR_STORE_NOT_READY_MESSAGE)

    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise VectorStoreNotReadyError(
            "chromadb is not installed. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
    collection_names = [collection.name for collection in client.list_collections()]
    if collection_name not in collection_names:
        raise VectorStoreNotReadyError(VECTOR_STORE_NOT_READY_MESSAGE)

    return client.get_collection(collection_name)


def normalize_query_type(classification_or_category: Any) -> str:
    """Extract a query type from a classification result or category string."""

    if isinstance(classification_or_category, dict):
        return str(classification_or_category.get("query_type", UNKNOWN_QUERY))
    if classification_or_category is None:
        return UNKNOWN_QUERY
    return str(classification_or_category)


def build_metadata_filter(query_type: str) -> dict[str, str] | None:
    """Build a Chroma metadata filter for the classified query type."""

    if query_type == PERSON_QUERY:
        return {"type": PERSON_QUERY}
    if query_type == PLACE_QUERY:
        return {"type": PLACE_QUERY}
    return None


def normalize_query_text(query: str) -> str:
    """Normalize query text for simple keyword hint matching."""

    normalized = re.sub(r"[^a-z0-9ğüşöçıİĞÜŞÖÇ]+", " ", query.lower())
    squashed = re.sub(r"\s+", " ", normalized).strip()
    return f" {squashed} "


def get_title_hints(query: str, classification: dict[str, Any]) -> list[str]:
    """Return exact Chroma title filters implied by the query."""

    titles: list[str] = []
    for title in classification.get("matched_people", []):
        if title not in titles:
            titles.append(title)
    for title in classification.get("matched_places", []):
        if title not in titles:
            titles.append(title)

    normalized_query = normalize_query_text(query)
    for keyword, hinted_titles in PLACE_HINTS_BY_KEYWORD.items():
        if f" {keyword} " in normalized_query:
            for title in hinted_titles:
                if title not in titles:
                    titles.append(title)

    return titles


def get_title_type(title: str) -> str | None:
    """Return the configured entity type for a known title."""

    if title in PEOPLE_TOPICS:
        return PERSON_QUERY
    if title in PLACE_TOPICS:
        return PLACE_QUERY
    return None


def title_filter(title: str) -> dict[str, Any]:
    """Build a Chroma filter for one known title."""

    title_type = get_title_type(title)
    if title_type is None:
        return {"title": title}
    return {"$and": [{"title": title}, {"type": title_type}]}


def query_collection(
    collection: Any,
    query_embedding: list[float],
    top_k: int,
    where_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one Chroma query and format the result."""

    query_args: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter is not None:
        query_args["where"] = where_filter

    return format_retrieval_results(collection.query(**query_args))


def deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate chunk ids while preserving order."""

    seen: set[str] = set()
    unique_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique_chunks.append(chunk)
    return unique_chunks


def format_retrieval_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a Chroma query response into stable retrieval records."""

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}
        retrieved.append(
            {
                "chunk_id": chunk_id,
                "title": metadata.get("title", ""),
                "type": metadata.get("type", ""),
                "source_url": metadata.get("source_url", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "word_count": metadata.get("word_count", 0),
                "char_count": metadata.get("char_count", 0),
                "text": documents[index],
                "distance": distances[index] if index < len(distances) else None,
            }
        )

    return retrieved


def retrieve_context(
    query: str,
    category: Any | None = None,
    top_k: int = config.DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Retrieve relevant chunks from Chroma using a local Ollama query embedding."""

    classification = category if category is not None else classify_query(query)
    query_type = normalize_query_type(classification)
    where_filter = build_metadata_filter(query_type)
    collection = get_chroma_collection()

    try:
        query_embedding = get_ollama_embedding(query)
    except OllamaUnavailableError as exc:
        raise OllamaUnavailableError(OLLAMA_NOT_REACHABLE_MESSAGE) from exc

    retrieved_chunks: list[dict[str, Any]] = []
    if isinstance(classification, dict):
        title_hints = get_title_hints(query, classification)
        if title_hints:
            base_count = max(1, top_k // len(title_hints))
            remainder = top_k % len(title_hints)
            for index, title in enumerate(title_hints):
                per_title_count = base_count + (1 if index < remainder else 0)
                retrieved_chunks.extend(
                    query_collection(
                        collection,
                        query_embedding,
                        per_title_count,
                        where_filter=title_filter(title),
                    )
                )

    if len(retrieved_chunks) < top_k:
        fallback_count = top_k + len(retrieved_chunks)
        retrieved_chunks.extend(
            query_collection(
                collection,
                query_embedding,
                fallback_count,
                where_filter=where_filter,
            )
        )

    return deduplicate_chunks(retrieved_chunks)[:top_k]


def retrieve_with_classification(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Classify a query and retrieve matching chunks."""

    classification = classify_query(query)
    chunks = retrieve_context(query, category=classification, top_k=top_k)
    return {
        "query": query,
        "classification": classification,
        "chunks": chunks,
    }


def format_sources(context_chunks: list[dict[str, object]]) -> list[str]:
    """Return source labels for retrieved context chunks."""

    sources: list[str] = []
    for chunk in context_chunks:
        title = chunk.get("title", "Unknown")
        chunk_index = chunk.get("chunk_index", "?")
        source_url = chunk.get("source_url", "")
        sources.append(f"{title} chunk {chunk_index}: {source_url}")
    return sources


def preview_text(text: str, max_length: int = 220) -> str:
    """Return a compact single-line text preview."""

    preview = " ".join(text.split())
    if len(preview) <= max_length:
        return preview
    return f"{preview[: max_length - 3]}..."


def print_retrieval_output(result: dict[str, Any]) -> None:
    """Print CLI retrieval output."""

    def print_line(message: str = "") -> None:
        encoding = sys.stdout.encoding or "utf-8"
        safe_message = message.encode(encoding, errors="replace").decode(encoding)
        print(safe_message)

    print_line(f"Query: {result['query']}")
    print_line("Classification result:")
    print_line(json.dumps(result["classification"], ensure_ascii=False, indent=2))
    print_line(f"Retrieved chunk count: {len(result['chunks'])}")

    for rank, chunk in enumerate(result["chunks"], start=1):
        distance = chunk["distance"]
        distance_text = f"{distance:.6f}" if isinstance(distance, (int, float)) else "n/a"
        print_line()
        print_line(f"Rank: {rank}")
        print_line(f"Title: {chunk['title']}")
        print_line(f"Type: {chunk['type']}")
        print_line(f"Chunk index: {chunk['chunk_index']}")
        print_line(f"Distance: {distance_text}")
        print_line(f"Source URL: {chunk['source_url']}")
        print_line(f"Preview: {preview_text(chunk['text'])}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Retrieve Wikipedia RAG context.")
    parser.add_argument("query", help="User query to classify and retrieve for.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help="Number of chunks to retrieve.",
    )
    return parser.parse_args()


def main() -> None:
    """Run classifier and retriever from the command line."""

    args = parse_args()
    try:
        result = retrieve_with_classification(args.query, top_k=args.top_k)
    except VectorStoreNotReadyError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    except OllamaUnavailableError as exc:
        print(str(exc))
        raise SystemExit(1) from exc

    print_retrieval_output(result)


if __name__ == "__main__":
    main()
