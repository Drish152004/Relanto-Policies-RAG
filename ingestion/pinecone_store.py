"""Upload child chunk embeddings to Pinecone."""

from __future__ import annotations

import logging

from pinecone import Pinecone, ServerlessSpec

from ingestion.chunker import ChildChunk
from utils.config import (
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    PINECONE_UPSERT_BATCH,
)

logger = logging.getLogger(__name__)


def get_index():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in environment")

    client = Pinecone(api_key=PINECONE_API_KEY)
    existing = {index.name for index in client.list_indexes()}

    if PINECONE_INDEX_NAME not in existing:
        logger.info("Creating Pinecone index: %s", PINECONE_INDEX_NAME)
        client.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    return client.Index(PINECONE_INDEX_NAME)


def _child_metadata(child: ChildChunk) -> dict:
    """Build Pinecone metadata for a child vector."""
    metadata = {
        "text": child.text,
        "parent_id": child.parent_id,
        "policy_name": child.policy_name,
        "section_title": child.section_title or "",
        "sub_section_title": child.sub_section_title or "",
        "page": child.page,
        "section_type": child.section_type,
        "source_file": child.source_file,
        "chunk_index": child.chunk_index,
    }
    return metadata


def upsert_children(
    children: list[ChildChunk],
    embeddings: list[list[float]],
) -> int:
    """Upsert child vectors to Pinecone. Returns vectors uploaded."""
    if len(children) != len(embeddings):
        raise ValueError("children and embeddings length mismatch")
    if not children:
        return 0

    index = get_index()
    uploaded = 0

    for start in range(0, len(children), PINECONE_UPSERT_BATCH):
        batch_children = children[start : start + PINECONE_UPSERT_BATCH]
        batch_embeddings = embeddings[start : start + PINECONE_UPSERT_BATCH]

        vectors = [
            {
                "id": child.child_id,
                "values": embedding,
                "metadata": _child_metadata(child),
            }
            for child, embedding in zip(batch_children, batch_embeddings)
        ]

        index.upsert(vectors=vectors)
        uploaded += len(vectors)
        logger.info("Upserted %s / %s vectors to Pinecone", uploaded, len(children))

    return uploaded
