"""End-to-end retrieval pipeline: routing -> filter -> search -> rerank -> parents."""

from __future__ import annotations

import logging

from retrieval.filter import build_pinecone_filter
from retrieval.cache import SemanticRetrievalCache, fingerprint_filter
from retrieval.guardrails import (
    extract_policy_keywords,
    is_query_in_rag_context,
    validate_query,
)
from retrieval.memory import ConversationMemory
from retrieval.parents import get_parent_contexts, keyword_search_parents
from retrieval.reranker import rerank_results
from retrieval.router import optimize_query, route_query
from retrieval.search import search_children

logger = logging.getLogger(__name__)


def retrieve_context(
    query: str,
    top_k: int = 5,
    candidate_pool_size: int = 20,
    force_source_files: list[str] | None = None,
    memory: ConversationMemory | None = None,
    semantic_cache: SemanticRetrievalCache | None = None,
) -> dict:
    """
    Execute the full retrieval flow for a user query.
    
    Args:
        query: Raw user query string.
        top_k: Final number of parent chunks to return.
        candidate_pool_size: Number of child chunks to retrieve from vector search
                             before reranking.
        force_source_files: Optional list of source files to override routing.
        
    Returns:
        dict: Containing optimized query details, child hits, and final parent contexts.
    """
    logger.info("Executing retrieval pipeline for query: %s", query)

    # 0. Execute AI Security Guardrail Validation Check with active session context
    document_metadata = ""
    if memory is not None:
        latest = memory.latest_turn()
        if latest:
            policy_hint = latest.semantic_intent or latest.effective_query or latest.user_query
            document_metadata = f"Active session conversation topic: {policy_hint}."

    guardrail = validate_query(query, document_metadata=document_metadata)
    if not guardrail["allowed"]:
        # Fallback is only allowed for low-risk relevance blocks, never for high-risk security blocks (like PII or injection attempts)
        if is_query_in_rag_context(query) and guardrail.get("risk_type") not in ("pii", "prompt_injection", "jailbreak", "sensitive_data", "harmful"):
            logger.info("Blocked query contains policy keywords. Redirecting to SQL Keyword search fallback.")
            kws = extract_policy_keywords(query)
            parent_contexts = keyword_search_parents(kws, limit=top_k)
            if parent_contexts:
                return {
                    "allowed": True,
                    "mode": "keyword_fallback",
                    "raw_query": query,
                    "optimized_query": "[SQL Keyword Search Fallback Mode (Security Enabled)]",
                    "keywords": kws,
                    "semantic_intent": f"Redirected to SQL keyword matching for terms: {kws}",
                    "source_files": [],
                    "child_hits": [],
                    "parent_contexts": parent_contexts,
                }
                
        logger.warning("Query blocked by guardrail: %s", guardrail["reason"])
        return {
            "allowed": False,
            "risk_type": guardrail["risk_type"],
            "reason": guardrail["reason"],
            "raw_query": query,
            "optimized_query": "",
            "keywords": [],
            "semantic_intent": "Blocked by AI safety guardrail",
            "source_files": [],
            "child_hits": [],
            "parent_contexts": [],
        }

    # Proceed using the sanitized query and resolve conversational follow-ups
    safe_query = guardrail["sanitized_query"]
    effective_query = safe_query
    memory_context = {"is_follow_up": False, "resolved_from_memory": False}
    if memory is not None:
        effective_query, memory_context = memory.resolve_follow_up(safe_query)

    # 1. Optimize the query using the LLM (and extract keywords & intent)
    optimization = optimize_query(effective_query)
    optimized_query = optimization["optimized_query"]
    keywords = optimization["keywords"]
    semantic_intent = optimization["semantic_intent"]

    # 2. Route the query to identify source document hints
    if force_source_files is not None:
        source_files = force_source_files
        logger.info("Overriding router. Using forced source files: %s", source_files)
    else:
        source_files = route_query(optimized_query, keywords)
        logger.info("Router identified matching source files: %s", source_files)

    # 3. Build the Pinecone metadata filter
    pinecone_filter = build_pinecone_filter(source_files)
    logger.info("Constructed Pinecone filter: %s", pinecone_filter)

    # 4. Reuse cached reranked retrievals before triggering vector search
    filter_key = fingerprint_filter(source_files)
    query_vector = None
    cache_meta = {
        "cache_hit": False,
        "cache_match_type": "disabled",
        "cache_confidence": 0.0,
        "embedding_reused": False,
    }
    if semantic_cache is not None:
        cached_result, cache_meta, query_vector = semantic_cache.lookup(
            query=optimized_query,
            filter_key=filter_key,
            candidate_pool_size=candidate_pool_size,
        )
        if cached_result is not None:
            cached_result.update(
                {
                    "raw_query": query,
                    "effective_query": effective_query,
                    "memory_context": memory_context,
                    "cache_hit": True,
                    "cache_match_type": cache_meta["cache_match_type"],
                    "cache_confidence": cache_meta["cache_confidence"],
                    "embedding_reused": cache_meta["embedding_reused"],
                }
            )
            if memory is not None:
                memory.add_turn(query, effective_query, cached_result)
            return cached_result

    # 5. Search Pinecone for child chunks (fetch a larger pool for reranking)
    child_hits = search_children(
        query=optimized_query,
        top_k=candidate_pool_size,
        pinecone_filter=pinecone_filter,
        query_vector=query_vector,
    )

    # 6. Rerank child chunks with BGE cross-encoder
    reranked_hits = rerank_results(
        query=effective_query,
        hits=child_hits,
        top_n=top_k,
    )

    # 7. Fetch parent texts from Neon PostgreSQL database
    parent_contexts = get_parent_contexts(reranked_hits)

    if not parent_contexts and keywords:
        logger.info("No parent contexts found. Retrying with keyword parent search.")
        parent_contexts = keyword_search_parents(keywords, limit=top_k)
        if parent_contexts:
            reranked_hits = []
            semantic_intent = f"{semantic_intent} (keyword retry used after low-confidence retrieval)"

    result = {
        "allowed": True,
        "raw_query": query,
        "effective_query": effective_query,
        "optimized_query": optimized_query,
        "keywords": keywords,
        "semantic_intent": semantic_intent,
        "source_files": source_files,
        "child_hits": reranked_hits,
        "parent_contexts": parent_contexts,
        "memory_context": memory_context,
        "cache_hit": False,
        "cache_match_type": cache_meta["cache_match_type"],
        "cache_confidence": cache_meta["cache_confidence"],
        "embedding_reused": cache_meta["embedding_reused"],
        "candidate_pool_size": candidate_pool_size,
    }

    if semantic_cache is not None:
        semantic_cache.store(
            raw_query=optimized_query,
            optimized_query=optimized_query,
            filter_key=filter_key,
            query_vector=query_vector,
            retrieval_result=result,
        )
    if memory is not None:
        memory.add_turn(query, effective_query, result)

    return result
