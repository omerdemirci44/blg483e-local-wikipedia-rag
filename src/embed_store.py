"""Local embedding generation and Chroma vector storage.

Sprint 3 reads processed chunks, generates embeddings through local Ollama, and
stores the vectors in a persistent Chroma collection. The module is import-safe:
no embedding work runs unless called explicitly or through ``python -m src.embed_store``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config


CHUNKS_PATH = config.PROCESSED_DATA_DIR / "chunks.jsonl"
OLLAMA_EMBEDDING_ENDPOINT = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
BATCH_SIZE = 50
REQUEST_TIMEOUT_SECONDS = 120

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "title",
    "type",
    "source_url",
    "text",
    "chunk_index",
    "word_count",
    "char_count",
}


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


def get_vector_store_path() -> Path:
    """Return the local directory for Chroma persistence."""

    return config.CHROMA_DB_DIR


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    """Load processed chunk records from the Sprint 2 JSONL file."""

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"chunk file not found: {chunks_path}. Run python -m src.chunker first."
        )

    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            chunk = json.loads(line)
            missing_fields = REQUIRED_CHUNK_FIELDS - set(chunk)
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"chunk line {line_number} is missing: {missing}")

            chunks.append(chunk)

    return chunks


def get_ollama_embedding(
    text: str,
    model: str = config.EMBEDDING_MODEL,
    endpoint: str = OLLAMA_EMBEDDING_ENDPOINT,
    session: Any | None = None,
) -> list[float]:
    """Generate one embedding vector using the local Ollama embeddings endpoint."""

    import requests

    client = session or requests.Session()
    payload = {
        "model": model,
        "prompt": text,
    }

    try:
        response = client.post(
            endpoint,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailableError(
            "Could not connect to Ollama at "
            f"{config.OLLAMA_BASE_URL}. Start Ollama and run: "
            f"ollama pull {config.EMBEDDING_MODEL}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaUnavailableError(
            "Timed out while calling Ollama. Confirm Ollama is running and run: "
            f"ollama pull {config.EMBEDDING_MODEL}"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Ollama embedding request failed with HTTP {response.status_code}: "
            f"{response.text}"
        ) from exc

    data = response.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("Ollama response did not include a non-empty embedding")

    return [float(value) for value in embedding]


def create_or_load_collection(collection_name: str = config.COLLECTION_NAME) -> Any:
    """Create or load the configured Chroma collection."""

    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "chromadb is not installed. Install project dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    config.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
    return client.get_or_create_collection(name=collection_name)


def reset_vector_store_if_needed(
    collection_name: str = config.COLLECTION_NAME,
    reset: bool = False,
) -> Any:
    """Reset the Chroma collection when requested, then return the collection."""

    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "chromadb is not installed. Install project dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    config.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))

    if reset:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    return client.get_or_create_collection(name=collection_name)


def build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Build Chroma metadata for one chunk."""

    return {
        "title": chunk["title"],
        "type": chunk["type"],
        "source_url": chunk["source_url"],
        "chunk_index": int(chunk["chunk_index"]),
        "word_count": int(chunk["word_count"]),
        "char_count": int(chunk["char_count"]),
    }


def upsert_embedding_batch(collection: Any, batch: list[dict[str, Any]]) -> None:
    """Upsert one batch of embedded chunks into Chroma."""

    collection.upsert(
        ids=[item["chunk_id"] for item in batch],
        documents=[item["text"] for item in batch],
        metadatas=[item["metadata"] for item in batch],
        embeddings=[item["embedding"] for item in batch],
    )


def get_existing_ids(
    collection: Any,
    ids: list[str],
    batch_size: int = 1000,
) -> set[str]:
    """Return ids that already exist in the Chroma collection."""

    existing_ids: set[str] = set()
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        result = collection.get(ids=batch_ids)
        existing_ids.update(result.get("ids", []))
    return existing_ids


def build_vector_store(
    chunks: list[dict[str, Any]],
    collection_name: str = config.COLLECTION_NAME,
    reset_collection: bool = False,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Generate embeddings for chunks and upsert them into Chroma."""

    import requests

    collection = reset_vector_store_if_needed(
        collection_name=collection_name,
        reset=reset_collection,
    )

    embedded_count = 0
    skipped_count = 0
    failures: list[dict[str, str]] = []
    batch: list[dict[str, Any]] = []
    existing_ids = set()
    if not reset_collection:
        existing_ids = get_existing_ids(
            collection,
            [str(chunk["chunk_id"]) for chunk in chunks],
        )

    with requests.Session() as session:
        for index, chunk in enumerate(chunks, start=1):
            if chunk["chunk_id"] in existing_ids:
                skipped_count += 1
                if index % 100 == 0:
                    print(f"Processed {index}/{len(chunks)} chunks...")
                continue

            try:
                embedding = get_ollama_embedding(chunk["text"], session=session)
                batch.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "metadata": build_metadata(chunk),
                        "embedding": embedding,
                    }
                )
                embedded_count += 1
            except OllamaUnavailableError:
                raise
            except Exception as exc:
                failures.append(
                    {
                        "chunk_id": str(chunk["chunk_id"]),
                        "error": str(exc),
                    }
                )

            if len(batch) >= batch_size:
                upsert_embedding_batch(collection, batch)
                batch = []

            if index % 100 == 0:
                print(f"Processed {index}/{len(chunks)} chunks...")

    if batch:
        upsert_embedding_batch(collection, batch)

    return {
        "chunks_embedded": embedded_count,
        "chunks_skipped": skipped_count,
        "collection_count": collection.count(),
        "collection_name": collection_name,
        "chroma_path": config.CHROMA_DB_DIR,
        "failures": failures,
    }


def embed_chunks(chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    """Generate embeddings for chunk records without writing them to Chroma."""

    import requests

    embedded_chunks: list[dict[str, object]] = []
    with requests.Session() as session:
        for chunk in chunks:
            embedding = get_ollama_embedding(str(chunk["text"]), session=session)
            embedded = dict(chunk)
            embedded["embedding"] = embedding
            embedded_chunks.append(embedded)

    return embedded_chunks


def store_embeddings(chunks_with_embeddings: list[dict[str, object]]) -> int:
    """Store already embedded chunks in Chroma and return the number stored."""

    collection = create_or_load_collection()
    batch: list[dict[str, Any]] = []

    for chunk in chunks_with_embeddings:
        batch.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": build_metadata(chunk),
                "embedding": chunk["embedding"],
            }
        )

    if batch:
        upsert_embedding_batch(collection, batch)

    return len(batch)


def run_embedding_pipeline(
    chunks_path: Path = CHUNKS_PATH,
    collection_name: str = config.COLLECTION_NAME,
    reset_collection: bool = False,
) -> dict[str, Any]:
    """Load chunks, embed them locally, store them in Chroma, and summarize."""

    chunks = load_chunks(chunks_path)
    vector_summary = build_vector_store(
        chunks,
        collection_name=collection_name,
        reset_collection=reset_collection,
    )

    return {
        "chunks_loaded": len(chunks),
        **vector_summary,
    }


def print_embedding_summary(summary: dict[str, Any]) -> None:
    """Print a clear command-line summary for Sprint 3."""

    print("Embedding and Chroma storage summary")
    print(f"Total chunks loaded: {summary['chunks_loaded']}")
    print(f"Total chunks embedded this run: {summary['chunks_embedded']}")
    print(f"Total chunks already in Chroma: {summary['chunks_skipped']}")
    print(f"Collection name: {summary['collection_name']}")
    print(f"Chroma path: {summary['chroma_path']}")
    print(f"Collection count: {summary['collection_count']}")

    failures = summary["failures"]
    if not failures:
        print("Failures: none")
        return

    print("Failures:")
    for failure in failures:
        print(f"- {failure['chunk_id']}: {failure['error']}")


def main() -> None:
    """Run the local embedding pipeline from the command line."""

    try:
        summary = run_embedding_pipeline()
    except OllamaUnavailableError as exc:
        print(str(exc))
        raise SystemExit(1) from exc

    print_embedding_summary(summary)
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
