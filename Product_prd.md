# Product PRD: Local Wikipedia RAG Assistant

## 1. Purpose

Build a local Retrieval Augmented Generation assistant that answers questions
about famous people and famous places using Wikipedia-derived content.

## 2. Problem Statement

Students need a small but complete RAG system that demonstrates ingestion,
manual chunking, local embeddings, vector retrieval, and local language model
generation without depending on hosted APIs.

## 3. Goals

- Run fully on localhost.
- Use Wikipedia content for at least 20 famous people and 20 famous places.
- Store all source data, processed chunks, and vector data locally.
- Generate embeddings locally.
- Use Chroma for vector storage.
- Use an Ollama-hosted local LLM for generation.
- Provide a simple Streamlit or CLI chat interface.
- Include clear documentation and recommendations.

## 4. Non-Goals

- No hosted LLM APIs.
- No production authentication or deployment workflow.
- No advanced web crawling beyond the selected Wikipedia pages.
- No full RAG implementation during Sprint 0.

## 5. Target Users

- University course staff evaluating the homework.
- Students or reviewers running the project locally.

## 6. Core User Flow

1. User installs dependencies.
2. User starts Ollama locally.
3. User ingests the selected Wikipedia pages.
4. System chunks documents and stores embeddings in Chroma.
5. User asks a question in the local interface.
6. System retrieves relevant chunks and generates a grounded answer.
7. System displays the answer with source context.

## 7. Functional Requirements

- Ingest Wikipedia documents for people and places.
- Validate that minimum topic counts are met.
- Manually chunk documents with metadata.
- Generate embeddings using a local model.
- Persist vectors in Chroma.
- Retrieve top-k relevant chunks for a query.
- Generate answers using a local Ollama LLM.
- Provide a simple local chat interface.

## 8. Quality Requirements

- Imports must not trigger ingestion, embedding, retrieval, or generation.
- The project should remain easy to inspect and run.
- The implementation should be modular and testable.
- Documentation should distinguish planned work from implemented work.

## 9. Data Requirements

- Raw Wikipedia article text will be stored in `data/raw/`.
- Processed chunks will be stored in `data/processed/`.
- Chroma vector data will be stored in `chroma_db/`.
- Each chunk should include article title, category, source, and chunk id.

## 10. Milestones

- Sprint 0: skeleton and documentation.
- Sprint 1: topic list and raw ingestion.
- Sprint 2: manual chunking.
- Sprint 3: local embeddings and Chroma storage.
- Sprint 4: classification and retrieval.
- Sprint 5: local generation.
- Sprint 6: interface and final polish.

## 11. Open Questions

- Which exact 20 people and 20 places will be selected?
- Should the interface be Streamlit, CLI, or both?
- Which prompt format will produce the clearest grounded answers?
- What evaluation questions should be included for the final report?
