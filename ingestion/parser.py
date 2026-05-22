"""Parse extracted PDF pages into hierarchical policy sections."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ingestion.loader import PageContent

# --- Heading patterns (Relanto policy layout) ---

TOC_LINE = re.compile(
    r"^(?:\d+\.\s+|\d+\.\d+(?:\.\d+)?\s+).+?\s*\.{2,}\s*\d+\s*$",
    re.IGNORECASE,
)

VERSION_LIKE = re.compile(
    r"^\d+\.\d+\s+(?:\d{1,2}\s+\w{3,9}|\w{3,9}\s+\d{2,4})",
    re.IGNORECASE,
)

SUB_HEADING = re.compile(
    r"^(\d+)\.(\d+)(?:\.(\d+))?\s+(.+?)(?:\s*\.{2,}\s*\d+)?\s*$",
    re.IGNORECASE,
)

MAIN_HEADING = re.compile(
    r"^(\d+)\.\s+(.+?)(?:\s*\.{2,}\s*\d+)?\s*$",
    re.IGNORECASE,
)

# Short title-case lines used by some decks (e.g. PMS objectives PDF).
UNNUMBERED_HEADING = re.compile(
    r"^[A-Z][A-Za-z0-9][A-Za-z0-9\s:&,\-/'()]{2,90}$",
)

FOOTER_MARKERS = (
    "confidential",
    "all rights reserved",
    "www.relanto.ai",
    "cin :",
    "cin:",
    "m: info@relanto",
    "relanto global private limited",
    "twelfth floor",
    "bannerghatta",
    "karnataka",
    "www.relanto",
    "copyright",
)

SKIP_PREFIXES = (
    "relanto global",
    "twelfth floor",
    "ibc knowledge",
    "bannerghatta",
    "bangalore",
    "www.relanto",
    "cin ",
    "m: info@",
    "version ",
    "©",
)

SECTION_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "table_of_contents": ("table of contents",),
    "metadata": (
        "document history",
        "author",
        "reviewer",
        "approver",
        "version",
        "cover",
    ),
    "annexure": ("annexure", "annex", "appendix"),
    "references": ("references", "reference"),
}


@dataclass
class Heading:
    level: int  # 1 = main, 2 = subsection, 3 = sub-subsection
    number: str
    title: str
    full: str


@dataclass
class _SectionBuffer:
    section_title: str | None = None
    sub_section_title: str | None = None
    section_type: str = "main_policy"
    start_page: int = 1
    content_lines: list[str] = field(default_factory=list)

    @property
    def heading_key(self) -> tuple[str | None, str | None]:
        return (self.section_title, self.sub_section_title)

    def content_text(self) -> str:
        return "\n".join(self.content_lines).strip()


def parse_document(
    pages: list[PageContent],
    source_filename: str,
) -> dict[str, Any]:
    """Parse all pages into structured sections for JSON export."""
    total_chars = sum(len(page.text) for page in pages)

    if total_chars == 0:
        return {
            "source_file": source_filename,
            "policy_name": _policy_name_from_filename(source_filename),
            "page_count": len(pages),
            "extraction_status": "no_text_layer",
            "extraction_note": (
                "No usable text after pdfplumber and OCR fallback."
            ),
            "sections": [],
        }

    policy_name = _extract_policy_title(pages) or _policy_name_from_filename(
        source_filename
    )
    sections: list[dict[str, Any]] = []
    buffer: _SectionBuffer | None = None
    toc_lines: list[str] = []
    in_toc = False
    current_main_title: str | None = None

    for page in pages:
        for line in page.lines:
            if _should_skip_line(line):
                continue

            if _is_table_of_contents_header(line):
                _flush_buffer(sections, buffer, policy_name)
                buffer = None
                in_toc = True
                toc_lines = [line]
                continue

            if in_toc:
                if _is_toc_line(line):
                    toc_lines.append(line)
                    continue
                _flush_toc(sections, policy_name, toc_lines, page.page_number)
                toc_lines = []
                in_toc = False

            heading = _parse_heading(line)
            if heading is None and _looks_like_unnumbered_heading(line):
                heading = Heading(
                    level=1,
                    number="",
                    title=line.strip(),
                    full=line.strip(),
                )

            if heading is not None:
                _flush_buffer(sections, buffer, policy_name)
                buffer = _start_buffer(
                    heading, page.page_number, current_main_title
                )
                if heading.level == 1:
                    current_main_title = heading.full
                elif current_main_title is None:
                    current_main_title = buffer.section_title
                continue

            if buffer is None:
                # Page 1 body is captured in the Cover section; skip orphan lines.
                if page.page_number == 1:
                    continue
                buffer = _SectionBuffer(
                    section_title="Preamble",
                    section_type="metadata",
                    start_page=page.page_number,
                )

            buffer.content_lines.append(line)

    if in_toc and toc_lines:
        _flush_toc(sections, policy_name, toc_lines, pages[-1].page_number)

    _flush_buffer(sections, buffer, policy_name)

    # Cover page metadata (page 1), kept separate from body sections.
    cover = _build_cover_section(pages, policy_name)
    if cover:
        sections.insert(0, cover)

    status = "ok" if sections else "partial"

    return {
        "source_file": source_filename,
        "policy_name": policy_name,
        "page_count": len(pages),
        "extraction_status": status,
        "sections": sections,
    }


def _parse_heading(line: str) -> Heading | None:
    """Detect numbered main/sub headings; ignore TOC dot-leader lines."""
    if _is_toc_line(line) or VERSION_LIKE.match(line):
        return None

    sub_match = SUB_HEADING.match(line)
    if sub_match:
        major, minor, subminor, title = sub_match.groups()
        title = title.strip()
        if not title or title[0].isdigit():
            return None
        if subminor:
            number = f"{major}.{minor}.{subminor}"
            level = 3
        else:
            number = f"{major}.{minor}"
            level = 2
        full = f"{number} {title}"
        return Heading(level=level, number=number, title=title, full=full)

    main_match = MAIN_HEADING.match(line)
    if main_match:
        major, title = main_match.groups()
        title = title.strip()
        if not title or title[0].isdigit() or "...." in title:
            return None
        number = major
        full = f"{number}. {title}"
        return Heading(level=1, number=number, title=title, full=full)

    return None


def _start_buffer(
    heading: Heading,
    page_number: int,
    current_main_title: str | None,
) -> _SectionBuffer:
    section_type = classify_section_type(heading.title)

    if heading.level == 1:
        return _SectionBuffer(
            section_title=heading.full,
            section_type=section_type,
            start_page=page_number,
        )

    major = heading.number.split(".")[0]
    main_title = current_main_title or f"{major}. {_infer_main_title(heading)}"

    return _SectionBuffer(
        section_title=main_title,
        sub_section_title=heading.full,
        section_type=section_type,
        start_page=page_number,
    )


def _infer_main_title(heading: Heading) -> str:
    """Fallback main title label when only a subsection heading is seen."""
    major = heading.number.split(".")[0]
    defaults = {
        "1": "Document History",
        "2": "Foreword",
        "3": "Guidelines",
        "4": "References",
        "5": "Exclusion",
        "6": "Non-Compliance",
        "7": "Revision of the Policy",
        "8": "Explanation/Deviation of Policy",
        "9": "Annexure",
    }
    return defaults.get(major, "Section")


def classify_section_type(title: str) -> str:
    """Map a section title to a semantic section type."""
    lowered = title.lower()

    for section_type, keywords in SECTION_TYPE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return section_type

    return "main_policy"


def _flush_buffer(
    sections: list[dict[str, Any]],
    buffer: _SectionBuffer | None,
    policy_name: str,
) -> None:
    if buffer is None:
        return

    content = buffer.content_text()
    if not content:
        return

    entry: dict[str, Any] = {
        "policy_name": policy_name,
        "section_title": buffer.section_title,
        "section_type": buffer.section_type,
        "page": buffer.start_page,
        "content": content,
    }
    if buffer.sub_section_title:
        entry["sub_section_title"] = buffer.sub_section_title

    if sections and _should_merge(sections[-1], entry):
        sections[-1]["content"] = (
            f"{sections[-1]['content']}\n\n{entry['content']}".strip()
        )
        return

    sections.append(entry)


def _should_merge(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Merge continuation pages that repeat the same heading."""
    return (
        previous.get("section_title") == current.get("section_title")
        and previous.get("sub_section_title") == current.get("sub_section_title")
        and previous.get("section_type") == current.get("section_type")
    )


def _flush_toc(
    sections: list[dict[str, Any]],
    policy_name: str,
    toc_lines: list[str],
    page_number: int,
) -> None:
    if not toc_lines:
        return
    sections.append(
        {
            "policy_name": policy_name,
            "section_title": "Table of Contents",
            "section_type": "table_of_contents",
            "page": page_number,
            "content": "\n".join(toc_lines),
        }
    )


def _build_cover_section(
    pages: list[PageContent],
    policy_name: str,
) -> dict[str, Any] | None:
    if not pages:
        return None

    cover_lines: list[str] = []
    for line in pages[0].lines:
        if _should_skip_line(line, cover_page=True):
            continue
        if line.lower() == "relanto":
            continue
        cover_lines.append(line)

    if not cover_lines:
        return None

    return {
        "policy_name": policy_name,
        "section_title": "Cover",
        "section_type": "metadata",
        "page": 1,
        "content": "\n".join(cover_lines),
    }


def _extract_policy_title(pages: list[PageContent]) -> str | None:
    """Read policy title from the first page (Relanto cover layout)."""
    if not pages:
        return None

    lines = [line for line in pages[0].lines if not _should_skip_line(line, cover_page=True)]
    title_parts: list[str] = []
    collecting = False

    for line in lines:
        lower = line.lower()
        if lower == "relanto":
            collecting = True
            continue
        if not collecting:
            continue
        if any(
            marker in lower
            for marker in ("version", "cin", "m: info", "www.", "©", "all rights")
        ):
            break
        if len(line) < 100:
            title_parts.append(line)

    if title_parts:
        return " ".join(title_parts).strip()

    for line in lines[:4]:
        if len(line) > 8 and not line.startswith("www."):
            return line

    return None


def _policy_name_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return stem.replace("_", " ").replace("  ", " ").strip()


def _is_table_of_contents_header(line: str) -> bool:
    return line.strip().lower() == "table of contents"


def _is_toc_line(line: str) -> bool:
    return bool(TOC_LINE.match(line.strip()))


def _should_skip_line(line: str, *, cover_page: bool = False) -> bool:
    stripped = line.strip()
    if not stripped:
        return True

    lower = stripped.lower()
    if any(marker in lower for marker in FOOTER_MARKERS):
        return True

    if cover_page and any(lower.startswith(prefix) for prefix in SKIP_PREFIXES):
        return True

    if len(stripped) <= 3 and stripped.isdigit():
        return True

    return False


def _looks_like_unnumbered_heading(line: str) -> bool:
    """Detect title-style headings without section numbers."""
    stripped = line.strip()
    if len(stripped) < 8 or len(stripped) > 90:
        return False
    if stripped[0] in "•●▪*-·":
        return False
    if stripped.endswith(".") and not stripped.endswith("..."):
        return False
    if _is_toc_line(stripped) or _parse_heading(stripped) is not None:
        return False
    if sum(1 for ch in stripped if ch.islower()) > len(stripped) * 0.6:
        return False

    words = stripped.split()
    if len(words) < 2:
        return False

    heading_hints = (
        "introduction",
        "objective",
        "policy",
        "timeline",
        "components",
        "guidelines",
        "setting",
        "pms",
        "performance",
        "training",
        "how to",
        "key ",
    )
    lowered = stripped.lower()
    if not any(hint in lowered for hint in heading_hints):
        return False

    return bool(UNNUMBERED_HEADING.match(stripped))
