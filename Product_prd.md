# Product Requirements Document: Local Wikipedia RAG Assistant

## Product Name

Local Wikipedia RAG Assistant

## Problem Statement

Students need a small, understandable RAG system that demonstrates the main
stages of retrieval augmented generation without relying on hosted LLM or
embedding APIs. The project must show ingestion, chunking, embeddings, vector
storage, retrieval, grounded generation, and a simple user interface.

## Goal

Build a simplified local assistant that answers questions about selected famous
people and famous places using Wikipedia-derived content. The system should run
on localhost, use local Ollama models, and expose both CLI and Streamlit access.

## Target Users

- Course instructors evaluating the homework.
- Students demonstrating a local RAG workflow.
- Reviewers who want to run the system on a local machine.

## User Stories

- As an instructor, I want to run one setup command so I can evaluate the full
  pipeline quickly.
- As a user, I want to ask a question about a famous person and receive a
  grounded answer from local Wikipedia data.
- As a user, I want to ask a question about a famous place and see the source
  context used.
- As a reviewer, I want unsupported questions to return `I don't know.` instead
  of unsupported general knowledge.
- As a student, I want modular code so each RAG stage is easy to inspect.

## Functional Requirements

- Ingest at least 20 famous people and 20 famous places from Wikipedia.
- Store raw article data as JSON under `data/raw/people/` and
  `data/raw/places/`.
- Clean common mojibake artifacts before chunking and embedding.
- Manually chunk documents without LangChain text splitters.
- Save chunks as JSON Lines under `data/processed/chunks.jsonl`.
- Generate embeddings locally with Ollama `nomic-embed-text`.
- Store vectors and metadata in one Chroma collection.
- Classify queries as `person`, `place`, `both`, or `unknown`.
- Retrieve relevant chunks from Chroma with metadata filtering.
- Generate answers locally with Ollama `llama3.2:3b`.
- Return sources with title, URL, and chunk index.
- Provide a Streamlit chat interface.
- Provide an end-to-end setup command through `python -m src.pipeline`.
- Provide lightweight pytest coverage for core modules.

## Non-Functional Requirements

- The system must run locally on the user's machine.
- No hosted LLM APIs or external embedding APIs should be used.
- Modules should be import-safe and avoid heavy work at import time.
- The implementation should remain simple enough for academic review.
- Generated data should stay out of Git.
- Setup and failure messages should be understandable.

## Data Requirements

- Dataset includes 20 famous people and 20 famous places.
- Each raw JSON record includes title, type, source URL, text, save timestamp,
  and word count.
- Each chunk includes chunk id, title, type, source URL, text, chunk index, word
  count, and character count.
- Chroma metadata includes title, type, source URL, chunk index, word count, and
  character count.

## Retrieval Requirements

- Query embeddings must be generated locally with `nomic-embed-text`.
- Person queries should filter Chroma metadata to `type = person`.
- Place queries should filter Chroma metadata to `type = place`.
- Mixed and unknown queries should use broad retrieval.
- Known entity mentions should favor chunks from matching titles.
- Retrieval results should include text, metadata, and distance when available.

## Generation Requirements

- Generation must use local Ollama `llama3.2:3b`.
- The prompt must instruct the model to answer only from retrieved context.
- The model must not be encouraged to use outside knowledge.
- If the context does not clearly support an answer, the answer should be exactly
  `I don't know.`
- Comparison answers should compare only facts present in retrieved context.

## UI Requirements

- Streamlit page title: Local Wikipedia RAG Assistant.
- Chat-style question input.
- Display generated answer.
- Display classification result.
- Display source titles, URLs, and chunk indexes.
- Include expandable retrieved context previews.
- Include example questions.
- Include a clear chat or reset button.

## Evaluation Criteria

- `python -m src.pipeline` prepares the local data and vector store.
- CLI generator answers supported questions with grounded content.
- Unsupported queries such as `Who is the president of Mars?` return
  `I don't know.`
- Streamlit app runs with `streamlit run src/app.py`.
- `python -m pytest tests/` passes or skips environment-dependent tests
  gracefully.

## MVP Scope

The MVP is a simplified local RAG assistant for a fixed Wikipedia dataset of
famous people and places. It includes local ingestion, chunking, embeddings,
retrieval, generation, UI, and tests.

## Out Of Scope

- Hosted deployment.
- User authentication.
- Production monitoring.
- Large-scale web crawling.
- Continuous Wikipedia synchronization.
- Advanced reranking or fine-tuned models.
- Multi-user persistence.
- LangChain integration.

## Success Metrics

- At least 40 Wikipedia entities are ingested.
- Chroma contains vector records for generated chunks.
- Example questions retrieve relevant sources.
- Grounded generation works for supported questions.
- Unsupported questions return `I don't know.`
- Tests pass on a prepared local environment.
