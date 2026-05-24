"""Lightweight query router and optimizer using Groq."""

from __future__ import annotations

import json
import logging

from llm.client import get_groq_client
from llm.prompts import QUERY_OPTIMIZATION_SYSTEM_PROMPT, QUERY_OPTIMIZATION_USER_TEMPLATE

logger = logging.getLogger(__name__)


def optimize_query(query: str) -> dict:
    """
    Call Groq to rewrite the query into an optimized semantic query,
    extract keywords, and determine semantic intent.
    
    Args:
        query: Raw user query.
        
    Returns:
        dict: Containing 'optimized_query', 'keywords', and 'semantic_intent'.
    """
    client = get_groq_client()
    system_prompt = QUERY_OPTIMIZATION_SYSTEM_PROMPT
    user_prompt = QUERY_OPTIMIZATION_USER_TEMPLATE.format(query=query)

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        
        data = json.loads(response.choices[0].message.content)
        
        # Standardize keys
        return {
            "optimized_query": data.get("optimized_query") or data.get("Optimized Retrieval Query") or query,
            "keywords": data.get("keywords") or data.get("Important Keywords") or [],
            "semantic_intent": data.get("semantic_intent") or data.get("Semantic Intent") or "Information retrieval",
        }
    except Exception as e:
        logger.error("Error optimizing query with Groq: %s", e)
        # Fallback values in case of API failure
        fallback_keywords = [w.strip("?,.!") for w in query.split() if len(w) > 3]
        return {
            "optimized_query": query,
            "keywords": fallback_keywords,
            "semantic_intent": "General query retrieval",
        }


def route_query(query: str, keywords: list[str]) -> list[str]:
    """
    Map query text and keywords to specific policy files to use for metadata filtering.
    
    Args:
        query: Raw query string.
        keywords: List of extracted keywords.
        
    Returns:
        list[str]: Source files matching the query intent.
    """
    query_lower = query.lower()
    keywords_lower = [k.lower() for k in keywords]

    # Keyword to source file mapping
    mapping = {
        "wfh": ["Relanto- Work From Home (WFH) Policy_Mar 26.pdf"],
        "work from home": ["Relanto- Work From Home (WFH) Policy_Mar 26.pdf"],
        "remote": ["Relanto- Work From Home (WFH) Policy_Mar 26.pdf"],
        
        "posh": ["POSH.pdf"],
        "harassment": ["POSH.pdf"],
        "sexual harassment": ["POSH.pdf"],
        
        "menstrual": ["Menstrual.pdf"],
        "period": ["Menstrual.pdf"],
        
        "insurance": ["Relanto - Insurance Policy_24-25.pdf"],
        "medical": ["Relanto - Insurance Policy_24-25.pdf"],
        "claim": ["Relanto - Insurance Policy_24-25.pdf"],
        
        "pms": ["Relanto - PMS_2024_objective settings.pdf", "PMS.pdf", "PMS H2.pdf"],
        "performance": ["Relanto - PMS_2024_objective settings.pdf", "PMS.pdf", "PMS H2.pdf"],
        "objective": ["Relanto - PMS_2024_objective settings.pdf", "PMS.pdf", "PMS H2.pdf"],
        "appraisal": ["Relanto - PMS_2024_objective settings.pdf", "PMS.pdf", "PMS H2.pdf"],
        
        "ai": ["ABC of AI Handbook_C1.pdf"],
        
        "security": ["Data Security.pdf"],
        "data security": ["Data Security.pdf"],
        "cyber": ["Data Security.pdf"],
        
        "handbook": ["Relanto Employee Handbook Jan 24 _ V 1.2_2025.pdf"],
        "employee handbook": ["Relanto Employee Handbook Jan 24 _ V 1.2_2025.pdf"],
    }

    matched_files = set()

    # Check for direct matches in raw query text
    for key, files in mapping.items():
        if key in query_lower:
            matched_files.update(files)

    # Check for matches in keywords
    for kw in keywords_lower:
        for key, files in mapping.items():
            if key in kw or kw in key:
                matched_files.update(files)

    return list(matched_files)
