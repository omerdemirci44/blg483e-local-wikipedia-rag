"""Answer generation placeholders.

The final assistant will use a local Ollama model and retrieved context to
generate grounded responses. Sprint 0 only defines import-safe interfaces.
"""

from __future__ import annotations


def build_prompt(question: str, context_chunks: list[dict[str, object]]) -> str:
    """Build a minimal placeholder prompt from a question and context."""

    # TODO: Sprint 5 - create a grounded prompt with citation instructions.
    _ = context_chunks
    return f"Question: {question}\nAnswer:"


def generate_answer(question: str, context_chunks: list[dict[str, object]]) -> str:
    """Return a placeholder answer without calling a local LLM."""

    # TODO: Sprint 5 - call Ollama and generate an answer from retrieved context.
    _ = build_prompt(question, context_chunks)
    return "Answer generation is not implemented yet."
