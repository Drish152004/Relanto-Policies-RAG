"""Generate stable parent_id / child_id values for chunk linking."""

from __future__ import annotations

import hashlib
import re


def _slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    return re.sub(r"[-\s]+", "-", value).strip("-") or "unknown"


def make_parent_id(
    policy_name: str,
    section_title: str | None,
    sub_section_title: str | None,
    page: int,
) -> str:
    """Deterministic parent id from policy + section identity + page."""
    key = "|".join(
        [
            _slug(policy_name),
            _slug(section_title or ""),
            _slug(sub_section_title or ""),
            str(page),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"p_{digest}"


def make_child_id(parent_id: str, chunk_index: int) -> str:
    """Deterministic child id from parent id and chunk position."""
    return f"{parent_id}_c{chunk_index:03d}"
