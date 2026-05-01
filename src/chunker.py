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
from .text_cleanup import find_mojibake_artifacts, cleanup_text


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

    cleaned = cleanup_text(text)
    return re.sub(r"\s+", " ", cleaned).strip()


def choose_chunk_end(text: str, start: int, max_end: int, chunk_size: int) -> int:
    """Choose a chunk end near a sentence or whitespace boundary."""

    if max_end >= len(text):
        return len(text)

    min_end = start + max(1, int(chunk_size * 0.6))
    min_end = min(min_end, max_end)

    sentence_window_start = max(min_end, max_end - 240)
    sentence_candidates = [
        text.rfind(boundary, sentence_window_start, max_end)
        for boundary in (". ", "! ", "? ", "; ")
    ]
    sentence_end = max(sentence_candidates)
    if sentence_end >= min_end:
        return sentence_end + 1

    whitespace_window_start = max(min_end, max_end - 120)
    whitespace_end = text.rfind(" ", whitespace_window_start, max_end)
    if whitespace_end >= min_end:
        return whitespace_end

    return max_end


def choose_chunk_start(text: str, start: int) -> int:
    """Move a chunk start to a nearby word boundary when possible."""

    if start <= 0 or start >= len(text):
        return max(0, min(start, len(text)))
    if text[start].isspace():
        return start + 1
    if text[start - 1].isspace():
        return start

    backward_limit = max(0, start - 40)
    previous_space = text.rfind(" ", backward_limit, start)
    if previous_space >= backward_limit:
        return previous_space + 1

    forward_limit = min(len(text), start + 40)
    next_space = text.find(" ", start, forward_limit)
    if next_space != -1:
        return next_space + 1

    return start


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
        max_end = min(start + chunk_size, text_length)
        end = choose_chunk_end(normalized, start, max_end, chunk_size)
        if end <= start:
            end = max_end

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        next_start = max(end - chunk_overlap, start + 1)
        adjusted_start = choose_chunk_start(normalized, next_start)
        start = adjusted_start if adjusted_start > start else next_start

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


def validate_chunks(
    chunks: list[dict[str, Any]],
    chunk_size: int = config.CHUNK_SIZE,
) -> dict[str, Any]:
    """Run lightweight quality checks on generated chunks."""

    empty_chunks = [chunk["chunk_id"] for chunk in chunks if not chunk["text"].strip()]
    oversized_chunks = [
        chunk["chunk_id"] for chunk in chunks if chunk["char_count"] > chunk_size
    ]
    mojibake_hits: dict[str, list[str]] = {}

    for chunk in chunks:
        artifacts = find_mojibake_artifacts(chunk["text"])
        if artifacts:
            mojibake_hits[chunk["chunk_id"]] = artifacts

    return {
        "empty_chunks": empty_chunks,
        "oversized_chunks": oversized_chunks,
        "mojibake_hits": mojibake_hits,
    }


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
    validation = validate_chunks(chunks, chunk_size=chunk_size)
    document_count = len(documents)
    average_chunks = len(chunks) / document_count if document_count else 0

    return {
        "documents_loaded": document_count,
        "chunks_created": len(chunks),
        "output_path": saved_path,
        "average_chunks_per_document": average_chunks,
        "validation": validation,
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
    validation = summary["validation"]
    issue_count = (
        len(validation["empty_chunks"])
        + len(validation["oversized_chunks"])
        + len(validation["mojibake_hits"])
    )
    print(f"Validation issues: {issue_count}")


def main() -> None:
    """Run document chunking from the command line."""

    summary = run_chunking()
    print_chunking_summary(summary)


if __name__ == "__main__":
    main()
