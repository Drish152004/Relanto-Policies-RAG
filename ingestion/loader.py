"""Load policy PDFs: pdfplumber first, OCR fallback for image-only PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from utils.config import DATA_RAW, MIN_TEXT_CHARS, OCR_DPI


@dataclass(frozen=True)
class PageContent:
    """Text extracted from a single PDF page (1-indexed page numbers)."""

    page_number: int
    text: str
    lines: list[str]


@dataclass(frozen=True)
class PdfLoadResult:
    """Pages plus which extraction path produced them."""

    pages: list[PageContent]
    extraction_method: str  # "pdfplumber" | "ocr"


def list_pdf_paths(raw_dir: Path | None = None) -> list[Path]:
    """Return sorted PDF paths from data/raw."""
    directory = raw_dir or DATA_RAW
    return sorted(directory.glob("*.pdf"))


def load_pdf(pdf_path: Path) -> PdfLoadResult:
    """
    Extract text from a PDF.

    Uses pdfplumber when a text layer exists; falls back to OCR otherwise.
    """
    pages = _load_with_pdfplumber(pdf_path)

    if _has_sufficient_text(pages):
        return PdfLoadResult(pages=pages, extraction_method="pdfplumber")

    try:
        from ingestion.ocr import ocr_pdf as _ocr_pdf

        ocr_pages = _ocr_pdf(pdf_path, dpi=OCR_DPI)
    except ImportError:
        return PdfLoadResult(pages=pages, extraction_method="pdfplumber")
    except Exception as exc:
        print(f"  OCR failed for {pdf_path.name}: {exc}")
        return PdfLoadResult(pages=pages, extraction_method="ocr")

    if _has_sufficient_text(ocr_pages):
        return PdfLoadResult(pages=ocr_pages, extraction_method="ocr")

    return PdfLoadResult(pages=ocr_pages, extraction_method="ocr")


def _load_with_pdfplumber(pdf_path: Path) -> list[PageContent]:
    """Extract text from each page via pdfplumber."""
    pages: list[PageContent] = []

    with pdfplumber.open(pdf_path) as document:
        for index, page in enumerate(document.pages, start=1):
            text = page.extract_text() or ""
            text = _normalize_text(text)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            pages.append(PageContent(page_number=index, text=text, lines=lines))

    return pages


def _has_sufficient_text(pages: list[PageContent]) -> bool:
    """True when extracted text is usable for the parser."""
    total_chars = sum(len(page.text) for page in pages)
    return total_chars >= MIN_TEXT_CHARS


def _normalize_text(text: str) -> str:
    """Fix common PDF encoding artifacts and normalize whitespace."""
    text = text.replace("\ufffd", "'")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text
