"""OCR fallback for image-only PDFs (pdf2image + pytesseract)."""

from __future__ import annotations

from pathlib import Path

from ingestion.loader import PageContent, _normalize_text
from utils.config import POPPLER_PATH, TESSERACT_CMD

try:
    import pytesseract
    from pdf2image import convert_from_path

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

except ImportError:
    pytesseract = None
    convert_from_path = None


def ocr_pdf(pdf_path: Path, *, dpi: int = 200) -> list[PageContent]:
    """
    Convert each PDF page to an image and extract text with Tesseract.

    Requires local installs: Poppler (pdf2image) and Tesseract OCR.
    """
    if convert_from_path is None or pytesseract is None:
        raise RuntimeError(
            "OCR dependencies missing. Install: pip install pdf2image pytesseract pillow"
        )

    if TESSERACT_CMD and pytesseract is not None:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    convert_kwargs: dict = {"dpi": dpi}
    if POPPLER_PATH:
        convert_kwargs["poppler_path"] = POPPLER_PATH

    images = convert_from_path(str(pdf_path), **convert_kwargs)
    pages: list[PageContent] = []

    for index, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image) or ""
        text = _normalize_text(text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        pages.append(PageContent(page_number=index, text=text, lines=lines))

    return pages
