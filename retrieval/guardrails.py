"""AI security and validation guardrail checks for user queries."""

from __future__ import annotations

import json
import logging

from llm.client import get_groq_client
from llm.prompts import GUARDRAIL_SYSTEM_PROMPT, GUARDRAIL_USER_TEMPLATE

logger = logging.getLogger(__name__)


def validate_query(query: str, document_metadata: str = "") -> dict:
    """
    Evaluate user query before retrieval for safety and relevance.
    
    Args:
        query: Raw user query.
        document_metadata: Metadata context or hints about uploaded documents.
        
    Returns:
        dict: Containing 'allowed' (bool), 'risk_type' (str), 'reason' (str), 
              and 'sanitized_query' (str).
    """
    logger.info("Executing guardrail validation check for query: %s", query)
    client = get_groq_client()

    system_prompt = GUARDRAIL_SYSTEM_PROMPT
    user_prompt = GUARDRAIL_USER_TEMPLATE.format(
        query=query,
        document_metadata=document_metadata,
    )

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
        
        result = {
            "allowed": bool(data.get("allowed", True)),
            "risk_type": str(data.get("risk_type", "none")),
            "reason": str(data.get("reason", "")),
            "sanitized_query": str(data.get("sanitized_query", query)),
        }
        logger.info("Guardrail check result: %s", result)
        return result
    except Exception as e:
        logger.error("Error executing guardrail validation: %s", e)
        # Default fallback to allow query so the system remains resilient
        return {
            "allowed": True,
            "risk_type": "none",
            "reason": f"Guardrail evaluation error: {e}",
            "sanitized_query": query,
        }


# Keywords related to the company's uploaded policy documents
POLICY_KEYWORDS = {
    "leave", "policy", "wfh", "work from home", "remote", "posh", "harassment",
    "menstrual", "period", "insurance", "medical", "claim", "pms", "performance",
    "appraisal", "objective", "handbook", "employee", "security", "data", "cyber", "ai"
}


def is_query_in_rag_context(query: str) -> bool:
    """
    Check if the user query contains terms referring to RAG policies.
    """
    q_lower = query.lower()
    return any(kw in q_lower for kw in POLICY_KEYWORDS)


def extract_policy_keywords(query: str) -> list[str]:
    """
    Extract matching policy key terms from the query.
    """
    q_lower = query.lower()
    found = []
    for kw in POLICY_KEYWORDS:
        if kw in q_lower:
            found.append(kw)
    return found

