from src.classifier import classify_query


def test_classifies_known_person_query():
    result = classify_query("Who was Albert Einstein?")

    assert result["query_type"] == "person"
    assert "Albert Einstein" in result["matched_people"]


def test_classifies_known_place_query():
    result = classify_query("Where is Hagia Sophia?")

    assert result["query_type"] == "place"
    assert "Hagia Sophia" in result["matched_places"]


def test_classifies_comparison_between_people_as_both():
    result = classify_query("Compare Albert Einstein and Nikola Tesla")

    assert result["query_type"] == "both"
    assert "Albert Einstein" in result["matched_people"]
    assert "Nikola Tesla" in result["matched_people"]


def test_classifies_person_topic_hint_query():
    result = classify_query("Which person is associated with electricity?")

    assert result["query_type"] == "person"
    assert "electricity" in result["person_keywords"]


def test_classifies_unsupported_query_as_unknown():
    result = classify_query("Who is the president of Mars?")

    assert result["query_type"] == "unknown"
    assert result["matched_people"] == []
    assert result["matched_places"] == []
