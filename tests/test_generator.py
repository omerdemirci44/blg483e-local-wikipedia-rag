import pytest

from src.embed_store import OllamaUnavailableError
from src.generator import GenerationModelUnavailableError, UNSUPPORTED_ANSWER, answer_query
from src.retriever import VectorStoreNotReadyError


def test_generator_returns_i_dont_know_for_obvious_unsupported_query():
    try:
        result = answer_query("Who is the president of Mars?")
    except VectorStoreNotReadyError:
        pytest.skip("Chroma vector store is not ready")
    except OllamaUnavailableError:
        pytest.skip("Ollama service is not reachable")
    except GenerationModelUnavailableError:
        pytest.skip("Ollama generation model is not available")

    assert result["answer"] == UNSUPPORTED_ANSWER
    assert result["chunks"] == []
    assert result["sources"] == []
