"""Session-scoped conversation memory for policy retrieval continuity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FOLLOW_UP_MARKERS = {
    "that",
    "this",
    "it",
    "its",
    "they",
    "them",
    "those",
    "above",
    "same",
    "section",
    "policy",
    "document",
}

STRONG_REFERENCE_MARKERS = {
    "that",
    "this",
    "it",
    "its",
    "they",
    "them",
    "those",
    "above",
    "same",
    "section",
    "policy",
    "document",
}


@dataclass
class ConversationTurn:
    """A compact record of a completed policy QA turn."""

    user_query: str
    effective_query: str
    source_files: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    semantic_intent: str = ""
    parent_contexts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversationMemory:
    """Tracks policy topics and source context for the active chat session."""

    max_turns: int = 8
    turns: list[ConversationTurn] = field(default_factory=list)

    def latest_turn(self) -> ConversationTurn | None:
        return self.turns[-1] if self.turns else None

    def add_turn(
        self,
        user_query: str,
        effective_query: str,
        retrieval_result: dict[str, Any],
    ) -> None:
        """Store only retrieval-safe metadata needed for continuity."""
        if not retrieval_result.get("allowed", True):
            return

        self.turns.append(
            ConversationTurn(
                user_query=user_query,
                effective_query=effective_query,
                source_files=list(retrieval_result.get("source_files") or []),
                keywords=list(retrieval_result.get("keywords") or []),
                semantic_intent=str(retrieval_result.get("semantic_intent") or ""),
                parent_contexts=list(retrieval_result.get("parent_contexts") or []),
            )
        )
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def resolve_follow_up(self, query: str) -> tuple[str, dict[str, Any]]:
        """
        Expand short follow-up references with the latest grounded policy topic.

        The returned metadata is diagnostic only. It does not expose hidden state,
        embeddings, or raw cache internals.
        """
        latest = self.latest_turn()
        if latest is None:
            return query, {"is_follow_up": False, "resolved_from_memory": False}

        normalized = query.lower().strip()
        words = {w.strip(" ?!.,:;()[]{}\"'").lower() for w in normalized.split()}
        has_marker = bool(words & FOLLOW_UP_MARKERS)
        has_strong_reference = bool(words & STRONG_REFERENCE_MARKERS)
        is_short = len(words) <= 8
        looks_like_fragment = normalized.startswith(("what about", "how about", "and ", "explain"))

        if not (has_marker or is_short or looks_like_fragment):
            return query, {"is_follow_up": False, "resolved_from_memory": False}

        policy_hint = latest.semantic_intent or latest.effective_query or latest.user_query
        source_hint = ", ".join(latest.source_files)
        source_clause = f" Source document focus: {source_hint}." if source_hint and has_strong_reference else ""
        resolved = (
            f"{query}\n\nConversation context: The previous policy topic was "
            f"{policy_hint}.{source_clause} If the current question names a new policy topic, prioritize the current question."
        )

        return resolved, {
            "is_follow_up": True,
            "resolved_from_memory": True,
            "previous_topic": policy_hint,
            "previous_sources": latest.source_files,
        }
