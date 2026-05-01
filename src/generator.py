"""Grounded answer generation with a local Ollama LLM."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import config
from .classifier import UNKNOWN_QUERY
from .embed_store import OllamaUnavailableError
from .retriever import (
    OLLAMA_NOT_REACHABLE_MESSAGE,
    VECTOR_STORE_NOT_READY_MESSAGE,
    VectorStoreNotReadyError,
    format_sources,
    retrieve_with_classification,
)


OLLAMA_GENERATE_ENDPOINT = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
GENERATION_TIMEOUT_SECONDS = 180
UNSUPPORTED_ANSWER = "I don't know."


class GenerationModelUnavailableError(RuntimeError):
    """Raised when the configured Ollama generation model is not available."""


def build_context(context_chunks: list[dict[str, Any]]) -> str:
    """Build a compact context block from retrieved chunks."""

    context_lines: list[str] = []
    for index, chunk in enumerate(context_chunks, start=1):
        context_lines.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Title: {chunk.get('title', '')}",
                    f"Type: {chunk.get('type', '')}",
                    f"Chunk index: {chunk.get('chunk_index', '')}",
                    f"URL: {chunk.get('source_url', '')}",
                    f"Text: {chunk.get('text', '')}",
                ]
            )
        )

    return "\n\n".join(context_lines)


def build_prompt(question: str, context_chunks: list[dict[str, Any]]) -> str:
    """Build a strict grounded-answer prompt for Ollama."""

    context = build_context(context_chunks)
    return f"""You are a local Wikipedia RAG assistant.
Answer only using the provided context.
Do not use outside knowledge.
If the answer is not clearly present in the context, say exactly "I don't know."
Keep answers concise but useful, usually 1-3 sentences.
When the context identifies who or what something is, summarize that identity with the key facts from the context.
Do not answer with only the entity name unless no other supported detail is present.
For comparison questions, compare only using the provided context.
For comparison questions, describe each entity separately first, then compare.
Do not claim two entities share a trait unless the context clearly states that trait for both.

Context:
{context}

Question: {question}

Answer:"""


def clean_model_answer(answer: str) -> str:
    """Normalize model output and enforce the exact unsupported answer string."""

    cleaned = answer.strip()
    if not cleaned:
        return UNSUPPORTED_ANSWER

    lowered = cleaned.lower().strip('"')
    if "i don't know" == lowered or "i do not know" == lowered:
        return UNSUPPORTED_ANSWER

    return cleaned


def call_ollama_generate(
    prompt: str,
    model: str = config.LLM_MODEL,
    endpoint: str = OLLAMA_GENERATE_ENDPOINT,
) -> str:
    """Call the local Ollama generation endpoint."""

    import requests

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.9,
        },
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=GENERATION_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailableError(OLLAMA_NOT_REACHABLE_MESSAGE) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaUnavailableError(OLLAMA_NOT_REACHABLE_MESSAGE) from exc
    except requests.exceptions.HTTPError as exc:
        response_text = response.text.lower()
        if response.status_code == 404 or "model" in response_text:
            raise GenerationModelUnavailableError(
                f"Generation model is not available. Run ollama pull {config.LLM_MODEL}."
            ) from exc
        raise RuntimeError(
            f"Ollama generation request failed with HTTP {response.status_code}: "
            f"{response.text}"
        ) from exc

    data = response.json()
    generated = data.get("response")
    if not isinstance(generated, str):
        raise RuntimeError("Ollama response did not include a text response")

    return clean_model_answer(generated)


def should_skip_generation(classification: dict[str, Any]) -> bool:
    """Return whether the query should be answered as unsupported without an LLM."""

    return classification.get("query_type") == UNKNOWN_QUERY


def generate_answer(
    question: str,
    context_chunks: list[dict[str, Any]] | None = None,
) -> str:
    """Generate an answer from provided retrieved chunks."""

    chunks = context_chunks or []
    if not chunks:
        return UNSUPPORTED_ANSWER

    prompt = build_prompt(question, chunks)
    return call_ollama_generate(prompt)


def answer_query(query: str, top_k: int = config.DEFAULT_TOP_K) -> dict[str, Any]:
    """Classify, retrieve, and generate a grounded answer for a query."""

    retrieval_result = retrieve_with_classification(query, top_k=top_k)
    classification = retrieval_result["classification"]
    chunks = retrieval_result["chunks"]

    if should_skip_generation(classification):
        answer = UNSUPPORTED_ANSWER
    else:
        answer = generate_answer(query, chunks)

    return {
        "query": query,
        "classification": classification,
        "answer": answer,
        "chunks": chunks,
        "sources": format_sources(chunks),
    }


def print_line(message: str = "") -> None:
    """Print text safely on Windows consoles with non-UTF-8 encodings."""

    encoding = sys.stdout.encoding or "utf-8"
    safe_message = message.encode(encoding, errors="replace").decode(encoding)
    print(safe_message)


def print_generation_output(result: dict[str, Any]) -> None:
    """Print CLI generation output."""

    print_line(f"Query: {result['query']}")
    print_line("Classification:")
    print_line(json.dumps(result["classification"], ensure_ascii=False, indent=2))
    print_line("Answer:")
    print_line(result["answer"])
    print_line("Sources:")

    if not result["chunks"]:
        print_line("- none")
        return

    for chunk in result["chunks"]:
        print_line(
            "- "
            f"{chunk['title']} "
            f"(chunk {chunk['chunk_index']}): "
            f"{chunk['source_url']}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate a local RAG answer.")
    parser.add_argument("query", help="User query to answer.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help="Number of retrieved chunks to use as context.",
    )
    return parser.parse_args()


def main() -> None:
    """Run local RAG answer generation from the command line."""

    args = parse_args()
    try:
        result = answer_query(args.query, top_k=args.top_k)
    except VectorStoreNotReadyError as exc:
        print_line(str(exc) or VECTOR_STORE_NOT_READY_MESSAGE)
        raise SystemExit(1) from exc
    except OllamaUnavailableError as exc:
        print_line(str(exc) or OLLAMA_NOT_REACHABLE_MESSAGE)
        raise SystemExit(1) from exc
    except GenerationModelUnavailableError as exc:
        print_line(str(exc))
        raise SystemExit(1) from exc

    print_generation_output(result)


if __name__ == "__main__":
    main()
