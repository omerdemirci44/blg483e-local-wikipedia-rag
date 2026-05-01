# BLG483E Local Wikipedia RAG Assistant

## Project Overview

This repository is a university homework project for building a fully local
Wikipedia Retrieval Augmented Generation assistant.

The finished system will ingest Wikipedia content about famous people and famous
places, manually chunk the documents, generate embeddings locally, store them in
Chroma, retrieve relevant context, and answer questions through a local Ollama
language model. The assistant will run on localhost and expose a simple
Streamlit or CLI chat interface.

Sprint 0 only prepares the project skeleton. The RAG pipeline is intentionally
not implemented yet.

## Planned Architecture

1. **Ingestion**: collect Wikipedia pages for at least 20 famous people and 20
   famous places into `data/raw/`.
2. **Chunking**: split each document manually into structured text chunks with
   source metadata.
3. **Embedding and Storage**: generate local embeddings and store chunks in a
   persistent Chroma collection under `chroma_db/`.
4. **Classification**: identify whether the user query is about a person, a
   place, or a mixed topic.
5. **Retrieval**: search Chroma for the most relevant chunks, optionally using
   metadata filters.
6. **Generation**: build a grounded prompt and call a local Ollama LLM.
7. **Interface**: provide a simple Streamlit or CLI chat experience.

## Planned Stack

- Python
- Wikipedia API client
- Ollama
- `nomic-embed-text` for local embeddings
- `llama3.2:3b` for local answer generation
- Chroma for vector storage
- Streamlit or CLI for the localhost interface
- pytest for basic validation

## Sprint Roadmap

- **Sprint 0**: prepare the skeleton, documentation, config constants, and
  import-safe placeholders.
- **Sprint 1**: choose famous people and places, collect Wikipedia documents,
  and save raw files locally.
- **Sprint 2**: implement manual chunking and save processed chunks.
- **Sprint 3**: generate local embeddings and persist them in Chroma.
- **Sprint 4**: implement query classification and retrieval.
- **Sprint 5**: connect Ollama generation with retrieved context.
- **Sprint 6**: build the Streamlit or CLI chat interface and polish docs.

## Basic Run Instructions

These commands are placeholders for later sprints:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.app
```

Before the final assistant can answer questions, Ollama must be running locally
and the planned models must be available:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Current Status

The repository contains a clean Sprint 0 skeleton. Python modules are
import-safe and contain TODO markers for future implementation work.
