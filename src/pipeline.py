"""High-level pipeline placeholders for the local RAG workflow."""

from __future__ import annotations

from .classifier import classify_query
from .generator import generate_answer
from .retriever import format_sources, retrieve_context


def answer_question(question: str) -> dict[str, object]:
    """Run the planned RAG pipeline with placeholder behavior."""

    # TODO: Sprint 6 - connect classification, retrieval, and generation.
    category = classify_query(question)
    context_chunks = retrieve_context(question, category=category)
    answer = generate_answer(question, context_chunks)
    sources = format_sources(context_chunks)
    return {
        "question": question,
        "category": category,
        "answer": answer,
        "sources": sources,
    }
