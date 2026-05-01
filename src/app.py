"""Streamlit chat interface for the local Wikipedia RAG assistant."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import GenerationModelUnavailableError, answer_query
from src.retriever import VectorStoreNotReadyError
from src.embed_store import OllamaUnavailableError


EXAMPLE_QUESTIONS = [
    "Who was Albert Einstein?",
    "What did Marie Curie discover?",
    "Where is Hagia Sophia?",
    "Which famous place is located in Turkey?",
    "Compare Albert Einstein and Nikola Tesla",
    "Who is the president of Mars?",
]

VECTOR_STORE_ERROR = "Vector store is not ready. Run python -m src.embed_store first."
OLLAMA_ERROR = "Ollama is not reachable. Make sure Ollama is running."
MODEL_ERROR = "Generation model is not available. Run ollama pull llama3.2:3b."


def initialize_session_state(st: Any) -> None:
    """Initialize Streamlit session state values."""

    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_chat(st: Any) -> None:
    """Clear the chat history."""

    st.session_state.messages = []


def context_preview(text: str, max_length: int = 500) -> str:
    """Return a compact context preview for the UI."""

    preview = " ".join(text.split())
    if len(preview) <= max_length:
        return preview
    return f"{preview[: max_length - 3]}..."


def build_error_result(message: str) -> dict[str, Any]:
    """Build a result object for setup/runtime errors."""

    return {
        "answer": message,
        "classification": None,
        "chunks": [],
        "sources": [],
        "error": True,
    }


def run_question(query: str) -> dict[str, Any]:
    """Run the RAG generator and convert common setup errors to UI messages."""

    try:
        result = answer_query(query)
        result["error"] = False
        return result
    except VectorStoreNotReadyError:
        return build_error_result(VECTOR_STORE_ERROR)
    except GenerationModelUnavailableError:
        return build_error_result(MODEL_ERROR)
    except OllamaUnavailableError:
        return build_error_result(OLLAMA_ERROR)


def render_sources(st: Any, chunks: list[dict[str, Any]]) -> None:
    """Render source metadata for retrieved chunks."""

    st.markdown("**Sources**")
    if not chunks:
        st.caption("No sources available.")
        return

    for chunk in chunks:
        title = chunk.get("title", "Unknown")
        chunk_index = chunk.get("chunk_index", "?")
        source_url = chunk.get("source_url", "")
        st.markdown(f"- [{title}]({source_url}) - chunk `{chunk_index}`")


def render_context(st: Any, chunks: list[dict[str, Any]]) -> None:
    """Render retrieved context previews in an expandable section."""

    with st.expander("Retrieved context previews"):
        if not chunks:
            st.caption("No retrieved context.")
            return

        for rank, chunk in enumerate(chunks, start=1):
            title = chunk.get("title", "Unknown")
            chunk_index = chunk.get("chunk_index", "?")
            chunk_type = chunk.get("type", "unknown")
            distance = chunk.get("distance")
            distance_text = (
                f"{distance:.4f}" if isinstance(distance, (int, float)) else "n/a"
            )
            st.markdown(
                f"**{rank}. {title}** | `{chunk_type}` | "
                f"chunk `{chunk_index}` | distance `{distance_text}`"
            )
            st.caption(context_preview(str(chunk.get("text", ""))))


def render_assistant_result(st: Any, result: dict[str, Any]) -> None:
    """Render one assistant response."""

    if result.get("error"):
        st.error(result["answer"])
        return

    st.markdown(result["answer"])

    classification = result.get("classification")
    if classification:
        with st.expander("Classification result"):
            st.json(classification)

    chunks = result.get("chunks", [])
    render_sources(st, chunks)
    render_context(st, chunks)


def render_chat_history(st: Any) -> None:
    """Render stored chat messages."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant_result(st, message["result"])


def render_sidebar(st: Any) -> str | None:
    """Render sidebar controls and return an example query if clicked."""

    st.sidebar.header("Examples")
    selected_query = None
    for question in EXAMPLE_QUESTIONS:
        if st.sidebar.button(question, use_container_width=True):
            selected_query = question

    st.sidebar.divider()
    if st.sidebar.button("Clear chat", use_container_width=True):
        reset_chat(st)
        st.rerun()

    st.sidebar.caption("Runs fully on localhost with Chroma and Ollama.")
    return selected_query


def process_query(st: Any, query: str) -> None:
    """Answer a user query and append it to chat history."""

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Retrieving context and generating answer..."):
        result = run_question(query)

    st.session_state.messages.append({"role": "assistant", "result": result})
    st.rerun()


def main() -> None:
    """Run the Streamlit chat app."""

    import streamlit as st

    st.set_page_config(
        page_title="Local Wikipedia RAG Assistant",
        layout="wide",
    )

    initialize_session_state(st)
    selected_query = render_sidebar(st)

    st.title("Local Wikipedia RAG Assistant")
    st.write(
        "Ask questions about the local Wikipedia collection of famous people and "
        "famous places. Answers are generated from locally retrieved context."
    )

    render_chat_history(st)

    typed_query = st.chat_input("Ask a question about the local Wikipedia data")
    query = selected_query or typed_query
    if query:
        process_query(st, query)


if __name__ == "__main__":
    main()
