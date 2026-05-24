"""Groq chat client wrapper."""

from __future__ import annotations

from groq import Groq
from utils.config import GROQ_API_KEY


def get_groq_client() -> Groq:
    """Initialize and return a Groq client instance."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the configuration")
    return Groq(api_key=GROQ_API_KEY)
