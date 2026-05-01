"""Manual document chunking placeholders.

Later sprints will split saved Wikipedia documents into overlapping chunks with
metadata. This module intentionally avoids external dependencies.
"""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Return placeholder chunks for a text document.

    Parameters are accepted now so callers can be wired before the real manual
    chunking strategy is implemented.
    """

    # TODO: Sprint 2 - implement deterministic manual text chunking.
    _ = (text, chunk_size, chunk_overlap)
    return []


def chunk_document(
    document_id: str,
    text: str,
    metadata: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Placeholder for turning one document into chunk records."""

    # TODO: Sprint 2 - include chunk ids, source title, category, and text.
    _ = (document_id, text, metadata)
    return []
