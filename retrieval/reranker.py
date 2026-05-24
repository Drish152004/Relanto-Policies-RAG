"""Rerank candidate child chunks using a BGE cross-encoder."""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder | None:
    """Lazy-load the BGE cross-encoder model."""
    global _reranker
    if _reranker is None:
        try:
            logger.info("Loading cross-encoder model BAAI/bge-reranker-base")
            _reranker = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        except Exception as e:
            logger.error("Failed to load cross-encoder: %s", e)
            _reranker = None
    return _reranker


def rerank_results(
    query: str,
    hits: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Rerank candidate child chunks with BGE cross-encoder.
    
    Args:
        query: User query string.
        hits: Retrieve candidate list of dicts.
        top_n: Number of final results to return.
        
    Returns:
        list[dict]: Reranked child chunk list of dicts.
    """
    if not hits:
        return []

    model = get_reranker()
    if model is None:
        logger.warning("Reranker model unavailable. Falling back to vector similarity scores.")
        # Sort by Pinecone score descending
        sorted_hits = sorted(hits, key=lambda x: x.get("score") or 0.0, reverse=True)
        return sorted_hits[:top_n]

    logger.info("Reranking %s candidates using cross-encoder", len(hits))
    pairs = [[query, hit["text"]] for hit in hits]
    
    try:
        scores = model.predict(pairs)
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)
            
        # Sort by cross-encoder score descending
        sorted_hits = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_hits[:top_n]
    except Exception as e:
        logger.error("Error during reranking calculation: %s. Falling back to vector scores.", e)
        sorted_hits = sorted(hits, key=lambda x: x.get("score") or 0.0, reverse=True)
        return sorted_hits[:top_n]
