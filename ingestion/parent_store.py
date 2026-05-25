"""Store parent chunks in Neon/Postgres."""

from __future__ import annotations

import logging

import psycopg2
from psycopg2.extras import execute_batch

from ingestion.chunker import ParentChunk
from utils.config import NEON_DATABASE_URL

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS parent_chunks (
    parent_id TEXT PRIMARY KEY,
    parent_text TEXT NOT NULL,
    policy_name TEXT NOT NULL,
    section_title TEXT,
    sub_section_title TEXT,
    page INTEGER NOT NULL,
    section_type TEXT NOT NULL,
    source_file TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

_INSERT_SQL = """
INSERT INTO parent_chunks (
    parent_id, parent_text, policy_name, section_title,
    sub_section_title, page, section_type, source_file
) VALUES (
    %(parent_id)s, %(parent_text)s, %(policy_name)s, %(section_title)s,
    %(sub_section_title)s, %(page)s, %(section_type)s, %(source_file)s
)
ON CONFLICT (parent_id) DO UPDATE SET
    parent_text = EXCLUDED.parent_text,
    policy_name = EXCLUDED.policy_name,
    section_title = EXCLUDED.section_title,
    sub_section_title = EXCLUDED.sub_section_title,
    page = EXCLUDED.page,
    section_type = EXCLUDED.section_type,
    source_file = EXCLUDED.source_file;
"""


def get_connection():
    if not NEON_DATABASE_URL:
        raise ValueError("NEON_DATABASE_URL is not set in environment")
    return psycopg2.connect(NEON_DATABASE_URL)


def init_schema() -> None:
    """Create parent_chunks table if missing."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        conn.commit()
    logger.info("Neon schema ready (parent_chunks)")


def upsert_parents(parents: list[ParentChunk]) -> int:
    """Insert or update parent chunks. Returns number of rows written."""
    if not parents:
        return 0

    rows = [
        {
            "parent_id": parent.parent_id,
            "parent_text": parent.parent_text,
            "policy_name": parent.policy_name,
            "section_title": parent.section_title,
            "sub_section_title": parent.sub_section_title,
            "page": parent.page,
            "section_type": parent.section_type,
            "source_file": parent.source_file,
        }
        for parent in parents
    ]

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_batch(cur, _INSERT_SQL, rows, page_size=200)
        conn.commit()

    logger.info("Inserted/updated %s parent chunks in Neon", len(rows))
    return len(rows)


def fetch_parent(parent_id: str) -> dict | None:
    """Fetch one parent chunk by id (for future retrieval)."""
    query = """
        SELECT parent_id, parent_text, policy_name, section_title,
               sub_section_title, page, section_type, source_file
        FROM parent_chunks
        WHERE parent_id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (parent_id,))
            row = cur.fetchone()

    if not row:
        return None

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
    return dict(zip(keys, row))


def fetch_parents_batch(parent_ids: list[str]) -> dict[str, dict]:
    """Fetch multiple parent chunks by their IDs in a single SQL batch query."""
    if not parent_ids:
        return {}

    query = """
        SELECT parent_id, parent_text, policy_name, section_title,
               sub_section_title, page, section_type, source_file
        FROM parent_chunks
        WHERE parent_id IN %s
    """
    
    results = {}
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

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (tuple(parent_ids),))
                rows = cur.fetchall()
                for row in rows:
                    parent_dict = dict(zip(keys, row))
                    results[row[0]] = parent_dict
    except Exception as e:
        logger.error("Error batch fetching parent chunks from PostgreSQL: %s", e)

    return results

