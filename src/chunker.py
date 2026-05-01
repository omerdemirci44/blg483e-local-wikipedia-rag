"""Manual document chunking for the local Wikipedia RAG assistant.

This module reads raw Wikipedia article JSON files and writes chunk records in
JSON Lines format. It is import-safe: chunking only runs when called explicitly
or when executing ``python -m src.chunker``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import config


CHUNKS_OUTPUT_PATH = config.PROCESSED_DATA_DIR / "chunks.jsonl"


def count_words(text: str) -> int:
    """Count words in a chunk of text."""

    return len(re.findall(r"\b\w+\b", text))


def slugify_title(title: str) -> str:
    """Convert a document title into the slug used in chunk ids."""

    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def normalize_text(text: str) -> str:
    """Normalize document text before splitting into character chunks."""

    return re.sub(r"\s+", " ", text).strip()


def load_raw_documents(
    people_dir: Path = config.RAW_PEOPLE_DIR,
    places_dir: Path = config.RAW_PLACES_DIR,
) -> list[dict[str, Any]]:
    """Load raw person and place documents from Sprint 1 JSON files."""

    documents: list[dict[str, Any]] = []
    for raw_dir in (people_dir, places_dir):
        if not raw_dir.exists():
            continue

        for path in sorted(raw_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as file:
                document = json.load(file)
            document["raw_path"] = str(path)
            documents.append(document)

    return documents


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character chunks.

    The overlap is measured in characters from the end of the previous chunk.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(normalized[start:end])

        if end == text_length:
            break

        start = end - chunk_overlap

    return chunks


def create_chunks_for_document(
    document: dict[str, Any],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Create chunk records for one raw Wikipedia document."""

    title = str(document["title"])
    document_type = str(document["type"])
    source_url = str(document["source_url"])
    slug = slugify_title(title)

    chunks = chunk_text(
        str(document.get("text", "")),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunk_records: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_records.append(
            {
                "chunk_id": f"{document_type}_{slug}_{chunk_index}",
                "title": title,
                "type": document_type,
                "source_url": source_url,
                "text": chunk,
                "chunk_index": chunk_index,
                "word_count": count_words(chunk),
                "char_count": len(chunk),
            }
        )

    return chunk_records


def save_chunks_jsonl(
    chunks: list[dict[str, Any]],
    output_path: Path = CHUNKS_OUTPUT_PATH,
) -> Path:
    """Save chunk records to a JSON Lines file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return output_path


def run_chunking(
    output_path: Path = CHUNKS_OUTPUT_PATH,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> dict[str, Any]:
    """Load raw documents, create chunks, save JSONL, and return a summary."""

    documents = load_raw_documents()
    chunks: list[dict[str, Any]] = []

    for document in documents:
        chunks.extend(
            create_chunks_for_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    saved_path = save_chunks_jsonl(chunks, output_path=output_path)
    document_count = len(documents)
    average_chunks = len(chunks) / document_count if document_count else 0

    return {
        "documents_loaded": document_count,
        "chunks_created": len(chunks),
        "output_path": saved_path,
        "average_chunks_per_document": average_chunks,
    }


def chunk_document(
    document_id: str,
    text: str,
    metadata: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Backward-compatible wrapper for older Sprint 0 callers."""

    metadata = metadata or {}
    document = {
        "title": metadata.get("title", document_id),
        "type": metadata.get("type", "document"),
        "source_url": metadata.get("source_url", ""),
        "text": text,
    }
    return create_chunks_for_document(document)


def print_chunking_summary(summary: dict[str, Any]) -> None:
    """Print a concise chunking summary for the command-line entry point."""

    print("Document chunking summary")
    print(f"Total raw documents loaded: {summary['documents_loaded']}")
    print(f"Total chunks created: {summary['chunks_created']}")
    print(f"Output path: {summary['output_path']}")
    print(
        "Average chunks per document: "
        f"{summary['average_chunks_per_document']:.2f}"
    )


def main() -> None:
    """Run document chunking from the command line."""

    summary = run_chunking()
    print_chunking_summary(summary)


if __name__ == "__main__":
    main()
