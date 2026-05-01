from src.text_cleanup import cleanup_text


def test_cleanup_text_removes_common_mojibake_artifacts():
    text = 'A â€“ B â€” C â€˜Dâ€™ â€œEâ€� ÂF'

    assert cleanup_text(text) == 'A – B — C \'D\' "E" F'


def test_cleanup_text_leaves_clean_text_unchanged():
    text = "Albert Einstein was a theoretical physicist."

    assert cleanup_text(text) == text
