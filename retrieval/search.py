"""Semantic top-k search over child chunks in Pinecone."""

from __future__ import annotations

import logging

from ingestion.embedder import embed_texts
from ingestion.pinecone_store import get_index

logger = logging.getLogger(__name__)


def search_children(
    query: str,
    top_k: int = 10,
    pinecone_filter: dict | None = None,
    query_vector: list[float] | None = None,
) -> list[dict]:
    """
    Search child chunks in Pinecone using semantic similarity.
    
    Args:
        query: Optimized query text.
        top_k: Number of candidate child chunks to retrieve.
        pinecone_filter: Pinecone metadata filter dict.
        
    Returns:
        list[dict]: List of child chunk match dictionaries with metadata and similarity scores.
    """
    if not query:
        return []

    if query_vector is None:
        logger.info("Embedding query: %s", query)
        query_vectors = embed_texts([query])
        if not query_vectors:
            logger.error("Failed to generate query embedding.")
            return []
        query_vector = query_vectors[0]

    # Get Pinecone index and query it
    index = get_index()
    logger.info("Querying Pinecone index with top_k=%s", top_k)
    try:
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter,
        )
    except Exception as e:
        logger.error("Error querying Pinecone index: %s", e)
        return []

    # Format the matches
    results = []
    for match in response.get("matches", []):
        meta = match.get("metadata") or {}
        results.append({
            "child_id": match.get("id"),
            "parent_id": meta.get("parent_id"),
            "text": meta.get("text"),
            "policy_name": meta.get("policy_name"),
            "section_title": meta.get("section_title"),
            "sub_section_title": meta.get("sub_section_title"),
            "page": int(meta.get("page", 0)) if meta.get("page") is not None else 0,
            "section_type": meta.get("section_type"),
            "source_file": meta.get("source_file"),
            "score": match.get("score", 0.0),
        })

    logger.info("Found %s matches from Pinecone", len(results))
    return results
