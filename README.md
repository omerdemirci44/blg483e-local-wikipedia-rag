# BLG483E Local Wikipedia RAG Assistant

## Overview

This project is a simplified local Retrieval Augmented Generation assistant for
a university homework. It answers questions about selected famous people and
famous places using Wikipedia-derived content stored and processed on the local
machine.

The system ingests Wikipedia pages, cleans and chunks article text, creates
local embeddings with Ollama, stores vectors in Chroma, retrieves relevant
context, and generates grounded answers with a local Ollama language model. No
OpenAI API, hosted LLM API, or external embedding API is used.

Demo video link: TODO

## What The System Does

- Ingests Wikipedia pages for 20 famous people and 20 famous places.
- Saves raw article JSON files under `data/raw/`.
- Cleans common encoding artifacts from scraped text.
- Manually chunks documents into JSON Lines under `data/processed/chunks.jsonl`.
- Generates local embeddings with `nomic-embed-text`.
- Stores vectors in Chroma under `chroma_db/`.
- Classifies queries as `person`, `place`, `both`, or `unknown`.
- Retrieves relevant chunks with metadata filtering.
- Generates grounded answers with `llama3.2:3b`.
- Returns exactly `I don't know.` for unsupported questions when the retrieved
  context does not support an answer.
- Provides both CLI commands and a Streamlit chat UI.

## Architecture

1. **Ingestion**: `src.ingest` fetches selected Wikipedia pages with `requests`
   and BeautifulSoup, extracts paragraph text, and saves JSON files.
2. **Cleanup and Chunking**: `src.text_cleanup` and `src.chunker` clean text and
   manually split it into overlapping chunks with metadata.
3. **Embedding and Storage**: `src.embed_store` calls local Ollama embeddings and
   upserts chunk vectors into a persistent Chroma collection.
4. **Classification**: `src.classifier` uses rule-based matching for known
   people, places, and comparison language.
5. **Retrieval**: `src.retriever` embeds the query locally, applies metadata
   filters, and returns relevant Chroma chunks.
6. **Generation**: `src.generator` builds a strict grounded prompt and calls the
   local Ollama generation endpoint.
7. **Interface**: `src.app` provides a Streamlit chat-style interface.
8. **Pipeline**: `src.pipeline` runs ingestion, chunking, and vector-store setup
   end to end.

## Local-Only Constraint

The project is designed to run on localhost. Wikipedia pages are fetched during
ingestion, but embeddings and answer generation are performed locally through
Ollama. Generated data and the vector store remain on the user's machine.

## Stack

- Python
- Ollama
- `nomic-embed-text`
- `llama3.2:3b`
- Chroma
- Streamlit
- pytest
- requests and BeautifulSoup

## Repository Structure

```text
src/
  app.py              Streamlit chat UI
  chunker.py          Manual chunking and chunk validation
  classifier.py       Rule-based query classification
  config.py           Project constants and paths
  embed_store.py      Ollama embeddings and Chroma storage
  generator.py        Local LLM answer generation
  ingest.py           Wikipedia ingestion
  pipeline.py         End-to-end setup pipeline
  retriever.py        Chroma retrieval
  text_cleanup.py     Text cleanup helpers
tests/
  test_*.py           Lightweight pytest suite
data/
  raw/                Generated raw Wikipedia JSON files
  processed/          Generated chunks JSONL
chroma_db/            Generated Chroma vector database
```

## Prerequisites

- Python
- Ollama
- Git

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Make sure Ollama is running before embedding, retrieval, or generation.

## Run Full Setup

This command ingests Wikipedia pages, chunks documents, and builds or updates the
Chroma vector store:

```powershell
python -m src.pipeline
```

Useful variants:

```powershell
python -m src.pipeline --skip-ingest
python -m src.pipeline --skip-ingest --skip-chunk
python -m src.pipeline --skip-embed
```

Use `--force` to rerun selected steps even when outputs already exist.

## Run Individual Steps

```powershell
python -m src.ingest
python -m src.chunker
python -m src.embed_store
```

## Run CLI Query Commands

Retriever only:

```powershell
python -m src.retriever "Who was Albert Einstein?"
```

Full retrieval plus local answer generation:

```powershell
python -m src.generator "Who was Albert Einstein?"
```

## Run Streamlit App

```powershell
streamlit run src/app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Run Tests

```powershell
python -m pytest tests/
```

The test suite is lightweight. Environment-dependent retrieval and generation
tests skip gracefully if Chroma or Ollama is unavailable.

## Example Queries

- Who was Albert Einstein?
- What did Marie Curie discover?
- Where is Hagia Sophia?
- Which famous place is located in Turkey?
- Compare Albert Einstein and Nikola Tesla
- Who is the president of Mars?

Unsupported questions should return:

```text
I don't know.
```

## Generated Data Notes

The following paths are generated locally and ignored by Git:

- `data/raw/`
- `data/processed/`
- `chroma_db/`

The `.gitkeep` files preserve the expected folder structure, but generated JSON,
JSONL, and Chroma files should not be committed.

## Troubleshooting

**Ollama is not reachable**

Start Ollama locally and rerun the command.

**Missing embedding model**

```powershell
ollama pull nomic-embed-text
```

**Missing generation model**

```powershell
ollama pull llama3.2:3b
```

**Vector store is not ready**

Run:

```powershell
python -m src.pipeline
```

or:

```powershell
python -m src.embed_store
```

**Streamlit import/path issue**

Run Streamlit from the repository root:

```powershell
streamlit run src/app.py
```

`src/app.py` includes a small project-root path bootstrap so `src.*` imports work
when Streamlit runs the file directly.
