"""Parent chunk creation, child splitting, and text cleaning."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    DATA_EXTRACTED,
    MIN_PARENT_CHARS,
    SKIP_SECTION_TYPES,
)
from utils.ids import make_child_id, make_parent_id

_DOT_LEADER = re.compile(r"\.{4,}")
_MULTI_SPACE = re.compile(r"[ \t]+")
_GARBAGE_LINE = re.compile(r"^[\s\.\-_=|•●▪·]{3,}$")


@dataclass(frozen=True)
class ParentChunk:
    parent_id: str
    parent_text: str
    policy_name: str
    section_title: str | None
    sub_section_title: str | None
    page: int
    section_type: str
    source_file: str


@dataclass(frozen=True)
class ChildChunk:
    child_id: str
    parent_id: str
    text: str
    policy_name: str
    section_title: str | None
    sub_section_title: str | None
    page: int
    section_type: str
    source_file: str
    chunk_index: int


def load_extracted_documents(extracted_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load all extracted JSON documents."""
    directory = extracted_dir or DATA_EXTRACTED
    documents: list[dict[str, Any]] = []

    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_json_path"] = str(path)
        documents.append(payload)

    return documents


def chunk_document(document: dict[str, Any]) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Build parent and child chunks from one extracted policy document."""
    source_file = document.get("source_file", "")
    policy_name = document.get("policy_name", source_file)
    sections = document.get("sections", [])

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    splitter = _build_splitter()

    for section in sections:
        parent = _section_to_parent(section, policy_name, source_file)
        if parent is None:
            continue

        parents.append(parent)
        child_texts = splitter.split_text(parent.parent_text)

        for index, child_text in enumerate(child_texts):
            cleaned = clean_text(child_text)
            if not cleaned:
                continue

            children.append(
                ChildChunk(
                    child_id=make_child_id(parent.parent_id, index),
                    parent_id=parent.parent_id,
                    text=cleaned,
                    policy_name=parent.policy_name,
                    section_title=parent.section_title,
                    sub_section_title=parent.sub_section_title,
                    page=parent.page,
                    section_type=parent.section_type,
                    source_file=parent.source_file,
                    chunk_index=index,
                )
            )

    return parents, children


def chunk_all_documents(
    documents: list[dict[str, Any]],
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Chunk every extracted document."""
    all_parents: list[ParentChunk] = []
    all_children: list[ChildChunk] = []

    for document in documents:
        if document.get("extraction_status") != "ok":
            continue

        parents, children = chunk_document(document)
        all_parents.extend(parents)
        all_children.extend(children)

    return all_parents, all_children


def clean_text(text: str) -> str:
    """Remove OCR noise, divider artifacts, and excess whitespace."""
    if not text:
        return ""

    cleaned_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _GARBAGE_LINE.match(line):
            continue
        if _DOT_LEADER.search(line) and len(line.replace(".", "").strip()) < 20:
            continue

        line = _DOT_LEADER.sub(" ", line)
        line = _MULTI_SPACE.sub(" ", line).strip()
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _section_to_parent(
    section: dict[str, Any],
    policy_name: str,
    source_file: str,
) -> ParentChunk | None:
    section_type = section.get("section_type", "main_policy")
    if section_type in SKIP_SECTION_TYPES:
        return None

    section_title = section.get("section_title")
    if section_title and section_title.strip().lower() == "cover":
        return None

    content = clean_text(section.get("content", ""))
    if len(content) < MIN_PARENT_CHARS:
        return None

    sub_section_title = section.get("sub_section_title")
    page = int(section.get("page", 0))

    parent_text = _build_parent_text(section_title, sub_section_title, content)
    parent_id = make_parent_id(policy_name, section_title, sub_section_title, page)

    return ParentChunk(
        parent_id=parent_id,
        parent_text=parent_text,
        policy_name=policy_name,
        section_title=section_title,
        sub_section_title=sub_section_title,
        page=page,
        section_type=section_type,
        source_file=source_file,
    )


def _build_parent_text(
    section_title: str | None,
    sub_section_title: str | None,
    content: str,
) -> str:
    heading = sub_section_title or section_title
    if heading:
        return f"{heading}\n\n{content}".strip()
    return content


def _build_splitter() -> RecursiveCharacterTextSplitter:
    encoding = tiktoken.get_encoding("cl100k_base")

    def token_length(text: str) -> int:
        return len(encoding.encode(text))

    return RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        length_function=token_length,
    )
