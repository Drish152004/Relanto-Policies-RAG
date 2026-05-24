"""Compose prompt with parent contexts + query; call LLM; return answer."""

from __future__ import annotations

import logging

from llm.client import get_groq_client
from llm.prompts import GROUNDED_QA_SYSTEM_PROMPT, GROUNDED_QA_USER_TEMPLATE

logger = logging.getLogger(__name__)


def generate_answer(
    query: str,
    contexts: list[dict],
    model: str = "llama-3.1-8b-instant",
) -> str:
    """
    Generate a grounded answer based on query and retrieved contexts.
    
    Args:
        query: User's question.
        contexts: List of dicts representing parent chunks, each containing
                  'parent_text', 'source_file', 'page', and 'section_title'.
        model: Groq chat model identifier.
        
    Returns:
        The generated answer string.
    """
    if not contexts:
        return "I'm sorry, but I couldn't find information regarding this in the company policy documents."

    client = get_groq_client()

    # Format the context blocks with source, page, and section title metadata
    formatted_contexts = []
    for i, ctx in enumerate(contexts):
        source = ctx.get("source_file") or ctx.get("policy_name") or "Unknown Policy"
        page = ctx.get("page", "Unknown")
        section = ctx.get("section_title") or "General"
        text = ctx.get("parent_text") or ""
        formatted_contexts.append(
            f"--- Context Segment {i + 1} ---\n"
            f"Source Document: {source}\n"
            f"Section: {section}\n"
            f"Page Number: {page}\n"
            f"Text Excerpt:\n{text}"
        )

    context_str = "\n\n".join(formatted_contexts)

    system_prompt = GROUNDED_QA_SYSTEM_PROMPT
    user_prompt = GROUNDED_QA_USER_TEMPLATE.format(context=context_str, query=query)

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("Error generating answer: %s", e)
        return f"An error occurred while generating the answer: {e}"
