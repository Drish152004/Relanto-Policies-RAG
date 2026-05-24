"""Fetch full parent chunks from Neon PostgreSQL based on child hits."""

from __future__ import annotations

import logging

from ingestion.parent_store import fetch_parent, get_connection

logger = logging.getLogger(__name__)


def get_parent_contexts(hits: list[dict]) -> list[dict]:
    """
    Fetch full parent text for each unique parent_id in child hits.
    
    Args:
        hits: List of reranked child chunk dicts.
        
    Returns:
        list[dict]: Deduplicated list of parent chunk dicts from PostgreSQL,
                    ordered by the priority of their highest-scoring child chunk.
    """
    if not hits:
        return []

    seen_parents = set()
    parent_contexts = []

    for hit in hits:
        parent_id = hit.get("parent_id")
        if not parent_id:
            logger.warning("Child chunk hit missing parent_id: %s", hit)
            continue
            
        if parent_id in seen_parents:
            continue

        logger.info("Fetching parent chunk from Postgres: %s", parent_id)
        try:
            parent = fetch_parent(parent_id)
            if parent:
                seen_parents.add(parent_id)
                # Keep track of the score that led to this parent retrieval
                parent["score"] = hit.get("rerank_score") or hit.get("score") or 0.0
                parent_contexts.append(parent)
            else:
                logger.warning("Parent chunk not found in Postgres: %s", parent_id)
        except Exception as e:
            logger.error("Error fetching parent chunk %s: %s", parent_id, e)

    return parent_contexts


def keyword_search_parents(keywords: list[str], limit: int = 3) -> list[dict]:
    """
    Query parent_chunks directly in PostgreSQL using a case-insensitive keyword search.
    Ranks the matching chunks by how many keywords they contain (relevance matching).
    
    Args:
        keywords: Clean policy keywords found in the user query.
        limit: Max parent chunks to return.
        
    Returns:
        list[dict]: Parent chunks matching the keywords, ordered by matches.
    """
    if not keywords:
        return []

    like_params = [f"%{kw}%" for kw in keywords]
    
    # Construct relevance scoring expression in SQL
    case_clauses = ["CASE WHEN parent_text ILIKE %s THEN 1 ELSE 0 END" for _ in keywords]
    case_expression = " + ".join(case_clauses)
    
    where_conditions = ["parent_text ILIKE %s" for _ in keywords]
    where_clause = " OR ".join(where_conditions)
    
    query = f"""
        SELECT parent_id, parent_text, policy_name, section_title,
               sub_section_title, page, section_type, source_file,
               ({case_expression}) as match_count
        FROM parent_chunks
        WHERE {where_clause}
        ORDER BY match_count DESC, parent_id ASC
        LIMIT %s
    """
    
    params = like_params + like_params + [limit]

    results = []
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()

        keys = [
            "parent_id",
            "parent_text",
            "policy_name",
            "section_title",
            "sub_section_title",
            "page",
            "section_type",
            "source_file",
        ]
        for row in rows:
            # Drop the match_count column when building parent dict
            parent = dict(zip(keys, row[:8]))
            parent["score"] = float(row[8])  # Score is match count
            results.append(parent)
    except Exception as e:
        logger.error("Error in database keyword parent search: %s", e)

    return results

