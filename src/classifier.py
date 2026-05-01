"""Query classification placeholders.

The assistant will eventually classify whether a user query is about a famous
person, a famous place, or a broader mixed request.
"""

from __future__ import annotations


UNKNOWN_CATEGORY = "unknown"


def classify_query(query: str) -> str:
    """Return a placeholder category for a user query."""

    # TODO: Sprint 4 - classify as person, place, mixed, or unknown.
    _ = query
    return UNKNOWN_CATEGORY


def is_supported_category(category: str, supported: tuple[str, ...]) -> bool:
    """Return whether a category is currently supported by the project."""

    return category in supported
