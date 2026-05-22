"""BGE embedding model for child chunks."""

from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

from utils.config import EMBED_BATCH_SIZE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-load the HuggingFace embedding model."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    """Embed a list of texts in batches."""
    if not texts:
        return []

    model = get_embeddings()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(model.embed_documents(batch))
        logger.info("Embedded %s / %s child chunks", min(start + batch_size, len(texts)), len(texts))

    return vectors
