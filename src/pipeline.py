"""End-to-end local setup pipeline for the Wikipedia RAG assistant.

The pipeline orchestrates the existing sprint modules without duplicating their
logic. It is import-safe: setup work only runs through ``python -m src.pipeline``
or an explicit function call.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import config
from .chunker import CHUNKS_OUTPUT_PATH, print_chunking_summary, run_chunking
from .embed_store import (
    OllamaUnavailableError,
    print_embedding_summary,
    run_embedding_pipeline,
)
from .generator import answer_query
from .ingest import (
    ingest_wikipedia_pages,
    ingestion_succeeded,
    print_ingestion_summary,
)


RAW_DATA_MISSING_MESSAGE = (
    "Raw data is missing. Run python -m src.ingest first or run pipeline "
    "without --skip-ingest."
)
CHUNKS_MISSING_MESSAGE = (
    "Chunks file is missing. Run python -m src.chunker first or run pipeline "
    "without --skip-chunk."
)


class PipelineSetupError(RuntimeError):
    """Raised for common setup pipeline failures."""


def raw_data_exists() -> bool:
    """Return whether raw people and places JSON files are available."""

    people_files = list(config.RAW_PEOPLE_DIR.glob("*.json"))
    place_files = list(config.RAW_PLACES_DIR.glob("*.json"))
    return bool(people_files) and bool(place_files)


def chunks_file_exists(chunks_path: Path = CHUNKS_OUTPUT_PATH) -> bool:
    """Return whether the processed chunks JSONL file exists."""

    return chunks_path.exists() and chunks_path.stat().st_size > 0


def count_chunks(chunks_path: Path = CHUNKS_OUTPUT_PATH) -> int:
    """Count non-empty lines in the chunks JSONL file."""

    if not chunks_file_exists(chunks_path):
        return 0

    with chunks_path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def chroma_collection_ready(collection_name: str = config.COLLECTION_NAME) -> bool:
    """Return whether Chroma exists and is current for the chunks file."""

    if not config.CHROMA_DB_DIR.exists():
        return False

    try:
        import chromadb
    except ModuleNotFoundError:
        return False

    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
        collection_names = [collection.name for collection in client.list_collections()]
        if collection_name not in collection_names:
            return False
        collection_count = client.get_collection(collection_name).count()
        expected_count = count_chunks()
        if expected_count > 0:
            return collection_count >= expected_count
        return collection_count > 0
    except Exception:
        return False


def run_ingestion_step(force: bool) -> dict[str, Any] | None:
    """Run or skip Wikipedia ingestion."""

    print("[1/3] Running Wikipedia ingestion...")
    if raw_data_exists() and not force:
        print("Raw data already exists; skipping ingestion. Use --force to rerun.")
        return None

    summary = ingest_wikipedia_pages()
    print_ingestion_summary(summary)
    if not ingestion_succeeded(summary):
        raise PipelineSetupError("Wikipedia ingestion did not complete successfully.")
    return summary


def run_chunking_step(skip_ingest: bool, force: bool) -> dict[str, Any] | None:
    """Run or skip document chunking."""

    print("[2/3] Running document chunking...")
    if not raw_data_exists():
        if skip_ingest:
            raise PipelineSetupError(RAW_DATA_MISSING_MESSAGE)
        raise PipelineSetupError("Raw data is missing after ingestion.")

    if chunks_file_exists() and not force:
        print("Chunks file already exists; skipping chunking. Use --force to rerun.")
        return None

    summary = run_chunking()
    print_chunking_summary(summary)
    if summary["chunks_created"] <= 0:
        raise PipelineSetupError("Document chunking created no chunks.")
    return summary


def run_embedding_step(skip_chunk: bool, force: bool) -> dict[str, Any] | None:
    """Run or skip embedding and Chroma vector store creation."""

    print("[3/3] Building Chroma vector store...")
    if not chunks_file_exists():
        if skip_chunk:
            raise PipelineSetupError(CHUNKS_MISSING_MESSAGE)
        raise PipelineSetupError("Chunks file is missing after chunking.")

    if chroma_collection_ready() and not force:
        print("Chroma collection already exists; skipping embedding. Use --force to rerun.")
        return None

    summary = run_embedding_pipeline(reset_collection=force)
    print_embedding_summary(summary)
    if summary["failures"]:
        raise PipelineSetupError("Embedding pipeline completed with failures.")
    return summary


def run_setup_pipeline(
    skip_ingest: bool = False,
    skip_chunk: bool = False,
    skip_embed: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Run selected setup steps in order and return step summaries."""

    summaries: dict[str, Any] = {
        "ingest": None,
        "chunk": None,
        "embed": None,
    }

    if skip_ingest:
        print("[1/3] Skipping Wikipedia ingestion by request.")
    else:
        summaries["ingest"] = run_ingestion_step(force=force)

    if skip_chunk:
        print("[2/3] Skipping document chunking by request.")
    else:
        summaries["chunk"] = run_chunking_step(
            skip_ingest=skip_ingest,
            force=force,
        )

    if skip_embed:
        print("[3/3] Skipping Chroma embedding by request.")
    else:
        summaries["embed"] = run_embedding_step(
            skip_chunk=skip_chunk,
            force=force,
        )

    print("Pipeline completed successfully.")
    return summaries


def answer_question(question: str) -> dict[str, Any]:
    """Backward-compatible wrapper for answering a question through Sprint 5."""

    return answer_query(question)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the local RAG setup pipeline.")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip Wikipedia ingestion and use existing raw data.",
    )
    parser.add_argument(
        "--skip-chunk",
        action="store_true",
        help="Skip document chunking and use existing chunks.jsonl.",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip Chroma embedding/vector store creation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected steps even if their outputs already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the setup pipeline from the command line."""

    args = parse_args()
    try:
        run_setup_pipeline(
            skip_ingest=args.skip_ingest,
            skip_chunk=args.skip_chunk,
            skip_embed=args.skip_embed,
            force=args.force,
        )
    except OllamaUnavailableError as exc:
        print(str(exc))
        raise SystemExit(1) from exc
    except PipelineSetupError as exc:
        print(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
