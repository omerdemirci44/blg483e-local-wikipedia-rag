"""Small text cleanup helpers shared by ingestion and chunking."""

from __future__ import annotations


MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u201c": "\u2013",
    "\u00e2\u20ac\u201d": "\u2014",
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\ufffd": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00c2": "",
}

MOJIBAKE_ARTIFACTS = tuple(MOJIBAKE_REPLACEMENTS)


def cleanup_text(text: str) -> str:
    """Replace common mojibake artifacts found in scraped Wikipedia text."""

    cleaned = text
    for artifact, replacement in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(artifact, replacement)
    return cleaned


def find_mojibake_artifacts(text: str) -> list[str]:
    """Return mojibake artifacts still present in text."""

    return [artifact for artifact in MOJIBAKE_ARTIFACTS if artifact in text]
