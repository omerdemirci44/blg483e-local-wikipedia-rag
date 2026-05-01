"""Project-wide configuration constants for the local Wikipedia RAG assistant.

Sprint 0 keeps configuration simple and import-safe. Later sprints can add
environment-variable overrides if the project needs them.
"""

from pathlib import Path


PROJECT_NAME = "Build a Local Wikipedia RAG Assistant"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_PEOPLE_DIR = RAW_DATA_DIR / "people"
RAW_PLACES_DIR = RAW_DATA_DIR / "places"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

PEOPLE_CATEGORY = "person"
PLACES_CATEGORY = "place"
SUPPORTED_CATEGORIES = (PEOPLE_CATEGORY, PLACES_CATEGORY)

MIN_PEOPLE_COUNT = 20
MIN_PLACES_COUNT = 20

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

COLLECTION_NAME = "wikipedia_people_places"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TOP_K = 4

# TODO: Sprint 2 - tune chunk size and overlap after sample ingestion.
# TODO: Sprint 3 - confirm local model names against the installed Ollama setup.
