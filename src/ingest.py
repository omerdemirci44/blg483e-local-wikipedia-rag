"""Wikipedia ingestion placeholders.

The final project will download or collect Wikipedia pages for famous people
and famous places. Sprint 0 only defines safe function boundaries for later
implementation.
"""

from __future__ import annotations

from pathlib import Path

from . import config


def get_seed_topics() -> dict[str, list[str]]:
    """Return the planned topic buckets for the initial Wikipedia dataset.

    The concrete list of at least 20 people and 20 places will be added in a
    later sprint.
    """

    # TODO: Sprint 1 - populate with at least 20 famous people and 20 places.
    return {
        "people": [],
        "places": [],
    }


def validate_seed_topics(topics: dict[str, list[str]]) -> bool:
    """Check whether the planned topic counts meet the project requirement."""

    people_count = len(topics.get("people", []))
    places_count = len(topics.get("places", []))
    return (
        people_count >= config.MIN_PEOPLE_COUNT
        and places_count >= config.MIN_PLACES_COUNT
    )


def ingest_wikipedia_pages(output_dir: Path = config.RAW_DATA_DIR) -> list[Path]:
    """Placeholder for collecting Wikipedia pages into the raw data directory."""

    # TODO: Sprint 1 - fetch or save Wikipedia article text files locally.
    _ = output_dir
    return []
