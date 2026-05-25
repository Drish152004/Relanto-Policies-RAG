"""In-memory semantic retrieval cache for session-level RAG reuse."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from ingestion.embedder import embed_texts

TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_query(query: str) -> str:
    """Normalize user text for exact cache lookup."""
    return " ".join(TOKEN_RE.findall(query.lower()))


def fingerprint_filter(source_files: list[str] | None) -> str:
    """Create a stable, non-sensitive key for retrieval filters."""
    if not source_files:
        return "all"
    joined = "\n".join(sorted(source_files))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def lexical_similarity(left: str, right: str) -> float:
    left_norm = normalize_query(left)
    right_norm = normalize_query(right)
    if not left_norm or not right_norm:
        return 0.0

    ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    return max(ratio, overlap)


@dataclass
class CacheEntry:
    raw_query: str
    optimized_query: str
    filter_key: str
    query_vector: list[float]
    retrieval_result: dict[str, Any]


@dataclass
class SemanticRetrievalCache:
    """Stores query-to-context mappings for one active conversation session."""

    similarity_threshold: float = 0.90
    lexical_threshold: float = 0.92
    max_entries: int = 32
    entries: list[CacheEntry] = field(default_factory=list)

    def lookup(
        self,
        query: str,
        filter_key: str,
        candidate_pool_size: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any], list[float] | None]:
        """
        Return a cached retrieval result when confidence is high.

        Exact and lexical checks avoid embedding generation. Embedding is created
        only when the cheap checks do not find a strong reusable context.
        """
        normalized = normalize_query(query)
        compatible = [
            entry
            for entry in self.entries
            if entry.filter_key == filter_key
            and int(entry.retrieval_result.get("candidate_pool_size", 0)) >= candidate_pool_size
        ]

        for entry in compatible:
            if normalize_query(entry.raw_query) == normalized:
                return self._mark_hit(entry, "exact", 1.0), {
                    "cache_hit": True,
                    "cache_match_type": "exact",
                    "cache_confidence": 1.0,
                    "embedding_reused": True,
                }, entry.query_vector

        best_lexical: tuple[CacheEntry, float] | None = None
        for entry in compatible:
            score = lexical_similarity(query, entry.raw_query)
            if best_lexical is None or score > best_lexical[1]:
                best_lexical = (entry, score)

        if best_lexical and best_lexical[1] >= self.lexical_threshold:
            entry, score = best_lexical
            return self._mark_hit(entry, "lexical", score), {
                "cache_hit": True,
                "cache_match_type": "lexical",
                "cache_confidence": score,
                "embedding_reused": True,
            }, entry.query_vector

        vectors = embed_texts([query])
        query_vector = vectors[0] if vectors else None
        if query_vector is None:
            return None, {
                "cache_hit": False,
                "cache_match_type": "miss",
                "cache_confidence": 0.0,
                "embedding_reused": False,
            }, None

        best_semantic: tuple[CacheEntry, float] | None = None
        for entry in compatible:
            score = cosine_similarity(query_vector, entry.query_vector)
            if best_semantic is None or score > best_semantic[1]:
                best_semantic = (entry, score)

        if best_semantic and best_semantic[1] >= self.similarity_threshold:
            entry, score = best_semantic
            return self._mark_hit(entry, "semantic", score), {
                "cache_hit": True,
                "cache_match_type": "semantic",
                "cache_confidence": score,
                "embedding_reused": False,
            }, query_vector

        return None, {
            "cache_hit": False,
            "cache_match_type": "miss",
            "cache_confidence": best_semantic[1] if best_semantic else 0.0,
            "embedding_reused": False,
        }, query_vector

    def store(
        self,
        raw_query: str,
        optimized_query: str,
        filter_key: str,
        query_vector: list[float] | None,
        retrieval_result: dict[str, Any],
    ) -> None:
        """Persist final reranked retrieval contexts for future reuse."""
        if not query_vector or not retrieval_result.get("parent_contexts"):
            return

        self.entries.append(
            CacheEntry(
                raw_query=raw_query,
                optimized_query=optimized_query,
                filter_key=filter_key,
                query_vector=query_vector,
                retrieval_result=dict(retrieval_result),
            )
        )
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    @staticmethod
    def _mark_hit(entry: CacheEntry, match_type: str, confidence: float) -> dict[str, Any]:
        result = dict(entry.retrieval_result)
        result["mode"] = "semantic_cache"
        result["cache_hit"] = True
        result["cache_match_type"] = match_type
        result["cache_confidence"] = confidence
        result["optimized_query"] = entry.optimized_query
        return result
