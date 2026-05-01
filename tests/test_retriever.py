import pytest

from src import config
from src.embed_store import OllamaUnavailableError
from src.retriever import VectorStoreNotReadyError, get_chroma_collection, retrieve_context


def test_retriever_returns_albert_einstein_when_vector_store_is_ready():
    try:
        collection = get_chroma_collection()
    except VectorStoreNotReadyError:
        pytest.skip("Chroma vector store is not ready")

    if collection.count() == 0:
        pytest.skip("Chroma vector store is empty")

    try:
        chunks = retrieve_context("Who was Albert Einstein?", top_k=config.DEFAULT_TOP_K)
    except OllamaUnavailableError:
        pytest.skip("Ollama embedding service is not reachable")

    assert chunks
    assert any(chunk["title"] == "Albert Einstein" for chunk in chunks)


def test_retriever_uses_topic_hint_for_electricity_question():
    try:
        collection = get_chroma_collection()
    except VectorStoreNotReadyError:
        pytest.skip("Chroma vector store is not ready")

    if collection.count() == 0:
        pytest.skip("Chroma vector store is empty")

    try:
        chunks = retrieve_context(
            "Which person is associated with electricity?",
            top_k=config.DEFAULT_TOP_K,
        )
    except OllamaUnavailableError:
        pytest.skip("Ollama embedding service is not reachable")

    assert chunks
    assert any(chunk["title"] == "Nikola Tesla" for chunk in chunks)
