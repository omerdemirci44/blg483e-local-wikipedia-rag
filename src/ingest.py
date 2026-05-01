"""Wikipedia ingestion for famous people and famous places.

This module fetches selected Wikipedia pages, extracts readable paragraph text,
and saves one JSON file per entity. It is import-safe: ingestion only runs when
called explicitly or when executing ``python -m src.ingest``.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from . import config
from .text_cleanup import cleanup_text


WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/"
REQUEST_TIMEOUT_SECONDS = 20
MIN_ARTICLE_WORDS = 100
USER_AGENT = (
    "BLG483E-Local-Wikipedia-RAG/1.0 "
    "(university homework; local ingestion script)"
)

PEOPLE_TOPICS = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    "Isaac Newton",
    "Charles Darwin",
    "Alan Turing",
    "Cleopatra",
    "Wolfgang Amadeus Mozart",
    "Pablo Picasso",
    "Nelson Mandela",
    "Abraham Lincoln",
    "Martin Luther King Jr.",
    "Queen Elizabeth II",
]

PLACE_TOPICS = [
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",
    "Mount Everest",
    "Louvre Museum",
    "Big Ben",
    "Stonehenge",
    "Petra",
    "Angkor Wat",
    "Sydney Opera House",
    "Burj Khalifa",
    "Niagara Falls",
    "Acropolis of Athens",
    "Mount Fuji",
]


def get_seed_topics() -> dict[str, list[str]]:
    """Return the configured Wikipedia topic buckets for ingestion."""

    return {
        "people": PEOPLE_TOPICS.copy(),
        "places": PLACE_TOPICS.copy(),
    }


def validate_seed_topics(topics: dict[str, list[str]]) -> bool:
    """Check whether the planned topic counts meet the project requirement."""

    people_count = len(topics.get("people", []))
    places_count = len(topics.get("places", []))
    return (
        people_count >= config.MIN_PEOPLE_COUNT
        and places_count >= config.MIN_PLACES_COUNT
    )


def safe_filename(title: str) -> str:
    """Convert a Wikipedia title into a safe lowercase JSON filename."""

    normalized = title.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    return f"{normalized}.json"


def build_wikipedia_url(title: str) -> str:
    """Build the English Wikipedia URL for an entity title."""

    article_slug = quote(title.replace(" ", "_"))
    return f"{WIKIPEDIA_BASE_URL}{article_slug}"


def count_words(text: str) -> int:
    """Count words in extracted article text."""

    return len(re.findall(r"\b\w+\b", text))


def clean_paragraph_text(text: str) -> str:
    """Remove citation markers and normalize whitespace from paragraph text."""

    without_citations = re.sub(r"\[\d+\]", "", text)
    normalized = re.sub(r"\s+", " ", without_citations).strip()
    return cleanup_text(normalized)


def extract_article_text(html: str) -> str:
    """Extract readable paragraph text from a Wikipedia article HTML page."""

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    parser_outputs = soup.select("div.mw-parser-output")
    if parser_outputs:
        content = max(parser_outputs, key=lambda element: len(element.find_all("p")))
    else:
        content = soup.select_one("main") or soup

    for selector in (
        "script",
        "style",
        "table",
        "nav",
        "aside",
        "figure",
        "sup.reference",
        "span.mw-editsection",
        "div.hatnote",
        "div.metadata",
        "div.navbox",
        "div.reflist",
        "div.references",
        "ol.references",
        "ul.gallery",
    ):
        for element in content.select(selector):
            element.decompose()

    paragraphs: list[str] = []
    for paragraph in content.find_all("p"):
        text = clean_paragraph_text(paragraph.get_text(" ", strip=True))
        if len(text) >= 40:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def fetch_wikipedia_article(title: str, session: Any | None = None) -> dict[str, Any]:
    """Fetch and extract one Wikipedia article."""

    import requests

    client = session or requests.Session()
    url = build_wikipedia_url(title)
    response = client.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    html = response.content.decode("utf-8", errors="replace")
    text = cleanup_text(extract_article_text(html))
    word_count = count_words(text)
    if word_count < MIN_ARTICLE_WORDS:
        raise ValueError(f"extracted only {word_count} words")

    return {
        "title": title,
        "source_url": response.url,
        "text": text,
        "word_count": word_count,
    }


def save_article_json(article: dict[str, Any], entity_type: str, output_dir: Path) -> Path:
    """Save one fetched article as a JSON file and return its path."""

    folder_name = "people" if entity_type == config.PEOPLE_CATEGORY else "places"
    target_dir = output_dir / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "title": article["title"],
        "type": entity_type,
        "source_url": article["source_url"],
        "text": article["text"],
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "word_count": article["word_count"],
    }

    file_path = target_dir / safe_filename(article["title"])
    file_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return file_path


def ingest_wikipedia_pages(output_dir: Path = config.RAW_DATA_DIR) -> dict[str, Any]:
    """Fetch configured Wikipedia pages and save them under the raw data folder."""

    import requests

    topics = get_seed_topics()
    if not validate_seed_topics(topics):
        raise ValueError("seed topics must include at least 20 people and 20 places")

    saved_files: list[Path] = []
    failed_pages: list[dict[str, str]] = []
    saved_counts = {
        config.PEOPLE_CATEGORY: 0,
        config.PLACES_CATEGORY: 0,
    }

    with requests.Session() as session:
        for entity_type, titles in (
            (config.PEOPLE_CATEGORY, topics["people"]),
            (config.PLACES_CATEGORY, topics["places"]),
        ):
            for title in titles:
                try:
                    article = fetch_wikipedia_article(title, session=session)
                    saved_files.append(save_article_json(article, entity_type, output_dir))
                    saved_counts[entity_type] += 1
                except Exception as exc:
                    failed_pages.append(
                        {
                            "title": title,
                            "type": entity_type,
                            "error": str(exc),
                        }
                    )

    return {
        "saved_files": saved_files,
        "people_saved": saved_counts[config.PEOPLE_CATEGORY],
        "places_saved": saved_counts[config.PLACES_CATEGORY],
        "failed_pages": failed_pages,
    }


def print_ingestion_summary(summary: dict[str, Any]) -> None:
    """Print a concise ingestion summary for the command-line entry point."""

    def print_line(message: str) -> None:
        encoding = sys.stdout.encoding or "utf-8"
        safe_message = message.encode(encoding, errors="replace").decode(encoding)
        print(safe_message)

    print_line("Wikipedia ingestion summary")
    print_line(f"Total people saved: {summary['people_saved']}")
    print_line(f"Total places saved: {summary['places_saved']}")

    failed_pages = summary["failed_pages"]
    if not failed_pages:
        print_line("Failed pages: none")
        return

    print_line("Failed pages:")
    for failure in failed_pages:
        print_line(f"- {failure['title']} ({failure['type']}): {failure['error']}")


def ingestion_succeeded(summary: dict[str, Any]) -> bool:
    """Return whether ingestion met the minimum project requirements."""

    return (
        summary["people_saved"] >= config.MIN_PEOPLE_COUNT
        and summary["places_saved"] >= config.MIN_PLACES_COUNT
        and not summary["failed_pages"]
    )


def main() -> None:
    """Run Wikipedia ingestion from the command line."""

    summary = ingest_wikipedia_pages()
    print_ingestion_summary(summary)
    if not ingestion_succeeded(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
