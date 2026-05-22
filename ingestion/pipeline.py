"""Extraction and indexing pipelines for policy PDFs."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ingestion.chunker import chunk_all_documents, load_extracted_documents
from ingestion.embedder import embed_texts
from ingestion.loader import list_pdf_paths, load_pdf
from ingestion.parent_store import init_schema, upsert_parents
from ingestion.pinecone_store import upsert_children
from ingestion.parser import parse_document
from utils.config import DATA_EXTRACTED, DATA_RAW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_all(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[dict]:
    """Extract every PDF in data/raw and save one JSON file per document."""
    raw_dir = raw_dir or DATA_RAW
    output_dir = output_dir or DATA_EXTRACTED
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for pdf_path in list_pdf_paths(raw_dir):
        load_result = load_pdf(pdf_path)
        document = parse_document(load_result.pages, pdf_path.name)
        document["extraction_method"] = load_result.extraction_method
        document["extracted_at"] = datetime.now(timezone.utc).isoformat()

        if (
            document["extraction_status"] == "no_text_layer"
            and load_result.extraction_method == "ocr"
        ):
            document["extraction_note"] = (
                "No text layer from pdfplumber; OCR fallback did not yield usable text. "
                "Check that Poppler and Tesseract are installed."
            )

        out_path = output_dir / f"{pdf_path.stem}.json"
        out_path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        results.append(
            {
                "source_file": pdf_path.name,
                "output_file": str(out_path),
                "extraction_status": document["extraction_status"],
                "extraction_method": document["extraction_method"],
                "sections": len(document["sections"]),
                "page_count": document["page_count"],
            }
        )

    return results


def index_all(extracted_dir: Path | None = None) -> dict:
    """
    Chunk extracted JSONs, embed children, store parents in Neon, vectors in Pinecone.
    """
    extracted_dir = extracted_dir or DATA_EXTRACTED
    documents = load_extracted_documents(extracted_dir)
    ok_documents = [doc for doc in documents if doc.get("extraction_status") == "ok"]

    logger.info("Loaded %s extracted documents (%s ok)", len(documents), len(ok_documents))

    parents, children = chunk_all_documents(ok_documents)
    logger.info("Created %s parent chunks", len(parents))
    logger.info("Created %s child chunks", len(children))

    if not parents:
        logger.warning("No parent chunks to index")
        return {"parents": 0, "children": 0, "neon_rows": 0, "pinecone_vectors": 0}

    init_schema()
    neon_rows = upsert_parents(parents)

    child_texts = [child.text for child in children]
    embeddings = embed_texts(child_texts)
    pinecone_vectors = upsert_children(children, embeddings)

    summary = {
        "documents": len(ok_documents),
        "parents": len(parents),
        "children": len(children),
        "neon_rows": neon_rows,
        "pinecone_vectors": pinecone_vectors,
    }
    logger.info("Indexing complete: %s", summary)
    return summary


def print_extract_summary(results: list[dict]) -> None:
    print(f"Extracted {len(results)} documents\n")
    for item in results:
        print(
            f"  {item['source_file']}: "
            f"{item['sections']} sections, "
            f"{item['page_count']} pages, "
            f"status={item['extraction_status']}, "
            f"method={item['extraction_method']}"
        )
        print(f"    -> {item['output_file']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy PDF extraction and indexing")
    parser.add_argument(
        "command",
        choices=["extract", "index"],
        help="extract: PDFs → JSON | index: JSON → Neon + Pinecone",
    )
    args = parser.parse_args()

    if args.command == "extract":
        results = extract_all()
        print_extract_summary(results)
    else:
        summary = index_all()
        print("\nIndexing summary")
        for key, value in summary.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
