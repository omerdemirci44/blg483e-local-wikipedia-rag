# Recommendation Skeleton

## Recommended Direction

Build the assistant as a small modular Python project with explicit stages for
ingestion, chunking, embedding, retrieval, generation, and interface code.

## Recommended Stack

- Use `wikipedia-api` or a similarly simple client for collecting article text.
- Use manual Python chunking instead of a framework-provided splitter.
- Use Ollama's local embedding model, planned as `nomic-embed-text`.
- Use Chroma as the local persistent vector database.
- Use Ollama's local generation model, planned as `llama3.2:3b`.
- Use Streamlit for a simple browser interface unless a CLI is preferred for
  simplicity.

## Data Recommendations

- Keep famous people and famous places in separate topic lists.
- Store raw article text before chunking so ingestion can be inspected.
- Store processed chunks with stable ids and metadata.
- Keep metadata simple: title, category, source URL, and chunk index.

## Implementation Recommendations

- Implement one pipeline stage per sprint.
- Keep each module import-safe.
- Avoid LangChain or other large orchestration frameworks unless the assignment
  later requires them.
- Prefer small functions that can be tested independently.
- Add tests around chunking, topic count validation, and retrieval formatting.

## Risk Notes

- Local model availability depends on the user's Ollama installation.
- Wikipedia article structure may vary across pages.
- Chunk size may need tuning after real documents are collected.
- Chroma persistence should be tested on a clean checkout before submission.

## Future Evaluation Ideas

- Prepare a small set of questions about both people and places.
- Check whether retrieved sources match the question topic.
- Verify that answers are grounded in retrieved chunks.
- Record known limitations and improvement ideas in the final report.
