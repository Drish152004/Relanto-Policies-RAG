# Relanto-Policies-RAG

Minimal production-style RAG over multiple policy PDFs: Streamlit, Groq, Pinecone (child vectors), Neon (parent chunks), parent-child chunking, metadata filtering, BGE embeddings + reranking.

## Project layout

```
RAG/
├── data/
│   ├── raw/              # Source policy PDFs (ingest input)
│   └── extracted/        # Structured JSON from PDF extraction
├── ingestion/
│   ├── loader.py         # pdfplumber + OCR fallback
│   ├── ocr.py            # pdf2image + pytesseract (image-only PDFs)
│   ├── parser.py         # Structured section parsing
│   ├── chunker.py        # Parent + child chunking
│   ├── embedder.py       # BGE embeddings (children only)
│   ├── pinecone_store.py # Child vectors → Pinecone
│   ├── parent_store.py   # Parent chunks → Neon/Postgres
│   └── pipeline.py       # extract | index commands
├── retrieval/
│   ├── router.py         # Optional query → metadata hints
│   ├── filter.py         # Chroma where-clause from metadata
│   ├── search.py         # Top-k child semantic search
│   ├── reranker.py       # BGE cross-encoder rerank
│   ├── parents.py        # parent_id → full parent context
│   └── pipeline.py       # Full retrieval flow
├── llm/
│   ├── client.py         # Groq client
│   ├── prompts.py        # Prompt templates
│   └── generate.py       # Context + query → answer
├── frontend/
│   └── app.py            # Streamlit UI
└── utils/
    ├── config.py         # Settings from .env
    └── ids.py            # Stable chunk IDs
```

## Retrieval flow

```
User Query → (router) → metadata filter → top-k children → rerank
  → parent_ids → load parents → LLM → answer
```

## Run

```bash
pip install -r req.txt

# Extract PDFs → data/extracted/*.json
python -m ingestion.pipeline extract

# Index: chunk → embed children → Pinecone + Neon parents
python -m ingestion.pipeline index

# .env required: PINECONE_API_KEY, NEON_DATABASE_URL
# Optional: PINECONE_INDEX_NAME (default: policy-rag)

# Full chat UI (after retrieval is implemented)
# streamlit run frontend/app.py
```

### OCR fallback (image-only PDFs)

Extraction uses **pdfplumber** first. If text is missing (`< 50` chars), it falls back to **OCR** (pdf2image + pytesseract). JSON includes `"extraction_method": "pdfplumber"` or `"ocr"`.

**Local installs required for OCR:**

| Tool | Purpose | Windows |
|------|---------|---------|
| [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) | PDF → images for pdf2image | Add `bin` folder to `PATH` |
| [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) | OCR engine | Install and add to `PATH` |

```bash
# Verify (optional)
tesseract --version
pdftoppm -v
```
