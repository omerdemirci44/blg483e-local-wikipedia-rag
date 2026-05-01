from src import config
from src.chunker import chunk_text, create_chunks_for_document


def test_chunk_text_creates_non_empty_chunks_with_size_limit():
    text = " ".join(f"word{i}" for i in range(120))

    chunks = chunk_text(text, chunk_size=120, chunk_overlap=30)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_chunk_text_overlap_roughly_preserves_context():
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa. "
        "Lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega. "
        "Second sentence gives the splitter enough room to create overlap."
    )

    chunks = chunk_text(text, chunk_size=90, chunk_overlap=25)

    assert len(chunks) > 1
    assert set(chunks[0].split()[-3:]) & set(chunks[1].split()[:6])


def test_create_chunks_for_document_preserves_metadata():
    document = {
        "title": "Test Person",
        "type": config.PEOPLE_CATEGORY,
        "source_url": "https://example.com/test-person",
        "text": " ".join(f"biography{i}" for i in range(80)),
    }

    chunks = create_chunks_for_document(document, chunk_size=140, chunk_overlap=30)
    first_chunk = chunks[0]

    assert first_chunk["chunk_id"] == "person_test_person_0"
    assert first_chunk["title"] == "Test Person"
    assert first_chunk["type"] == "person"
    assert first_chunk["source_url"] == "https://example.com/test-person"
    assert first_chunk["chunk_index"] == 0
    assert first_chunk["word_count"] > 0
    assert first_chunk["char_count"] == len(first_chunk["text"])
