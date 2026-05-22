"""Project paths and settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_EXTRACTED = PROJECT_ROOT / "data" / "extracted"

# Extraction
MIN_TEXT_CHARS = 50
OCR_DPI = 200
POPPLER_PATH = os.getenv("POPPLER_PATH")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

# Chunking
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 30
MIN_PARENT_CHARS = 40
SKIP_SECTION_TYPES = ("table_of_contents",)

# Embeddings
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768
EMBED_BATCH_SIZE = 32

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "policy-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_UPSERT_BATCH = 100

# Neon / Postgres
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
