"""Rule-based query classification for the local Wikipedia RAG assistant."""

from __future__ import annotations

import re
from typing import Any

from . import config
from .ingest import PEOPLE_TOPICS, PLACE_TOPICS


PERSON_QUERY = config.PEOPLE_CATEGORY
PLACE_QUERY = config.PLACES_CATEGORY
BOTH_QUERY = "both"
UNKNOWN_QUERY = "unknown"

PERSON_KEYWORDS = {
    "biography",
    "born",
    "died",
    "life",
    "invented",
    "discovered",
    "wrote",
    "scientist",
    "artist",
    "author",
    "poet",
    "footballer",
    "singer",
    "painter",
    "leader",
    "king",
    "queen",
}

PLACE_KEYWORDS = {
    "where",
    "place",
    "located",
    "location",
    "country",
    "city",
    "built",
    "architecture",
    "landmark",
    "monument",
    "museum",
    "mountain",
    "tower",
    "ancient",
    "turkey",
    "china",
    "india",
    "italy",
    "france",
    "jordan",
    "australia",
    "japan",
    "greece",
}

COMPARISON_KEYWORDS = {
    "compare",
    "comparison",
    "versus",
    "vs",
    "difference",
    "differences",
    "similarity",
    "similarities",
    "between",
}


def normalize_text(text: str) -> str:
    """Normalize text for simple rule matching."""

    lowered = text.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", normalized).strip()


def entity_aliases(entity: str) -> set[str]:
    """Return simple aliases for an entity name."""

    normalized = normalize_text(entity)
    parts = normalized.split()
    aliases = {normalized}
    special_aliases = {
        "martin luther king jr": {"martin luther king", "mlk"},
        "queen elizabeth ii": {"queen elizabeth"},
        "great wall of china": {"great wall"},
        "pyramids of giza": {"pyramids", "giza pyramids"},
        "acropolis of athens": {"acropolis"},
        "louvre museum": {"louvre"},
        "mount fuji": {"fuji"},
        "mount everest": {"everest"},
    }

    if len(parts) > 1:
        aliases.add(parts[-1])

    aliases.update(special_aliases.get(normalized, set()))

    return {alias for alias in aliases if alias}


def find_entity_matches(query: str, entities: list[str]) -> list[str]:
    """Find known entities mentioned in a query."""

    normalized_query = f" {normalize_text(query)} "
    matches: list[str] = []

    for entity in entities:
        for alias in entity_aliases(entity):
            if f" {alias} " in normalized_query:
                matches.append(entity)
                break

    return matches


def find_keyword_matches(query: str, keywords: set[str]) -> list[str]:
    """Find keyword matches in normalized query text."""

    normalized_query = f" {normalize_text(query)} "
    return sorted(keyword for keyword in keywords if f" {keyword} " in normalized_query)


def classify_query(query: str) -> dict[str, Any]:
    """Classify a user query as person, place, both, or unknown."""

    matched_people = find_entity_matches(query, PEOPLE_TOPICS)
    matched_places = find_entity_matches(query, PLACE_TOPICS)
    person_keywords = find_keyword_matches(query, PERSON_KEYWORDS)
    place_keywords = find_keyword_matches(query, PLACE_KEYWORDS)
    comparison_keywords = find_keyword_matches(query, COMPARISON_KEYWORDS)

    has_person_signal = bool(matched_people or person_keywords)
    has_place_signal = bool(matched_places or place_keywords)

    if matched_people and matched_places:
        query_type = BOTH_QUERY
        reason = "query mentions known people and known places"
    elif comparison_keywords and len(matched_people) + len(matched_places) >= 2:
        query_type = BOTH_QUERY
        reason = "query compares multiple known entities"
    elif has_person_signal and has_place_signal and comparison_keywords:
        query_type = BOTH_QUERY
        reason = "query has mixed comparison signals"
    elif has_person_signal and not has_place_signal:
        query_type = PERSON_QUERY
        reason = "query has person entity or person keyword matches"
    elif has_place_signal and not has_person_signal:
        query_type = PLACE_QUERY
        reason = "query has place entity or place keyword matches"
    elif has_person_signal and has_place_signal:
        query_type = BOTH_QUERY
        reason = "query has both person and place signals"
    else:
        query_type = UNKNOWN_QUERY
        reason = "no known entity or clear keyword signal found"

    return {
        "query_type": query_type,
        "matched_people": matched_people,
        "matched_places": matched_places,
        "person_keywords": person_keywords,
        "place_keywords": place_keywords,
        "comparison_keywords": comparison_keywords,
        "reason": reason,
    }


def is_supported_category(category: str, supported: tuple[str, ...]) -> bool:
    """Return whether a category is currently supported by the project."""

    return category in supported or category in {BOTH_QUERY, UNKNOWN_QUERY}
