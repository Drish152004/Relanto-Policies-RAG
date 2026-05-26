import os
import sys
import logging
import warnings
from pathlib import Path
from typing import List, Optional

# Suppress console clutter warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from retrieval.cache import SemanticRetrievalCache
from retrieval.pipeline import retrieve_context
from llm.generate import generate_answer
from utils.config import DATA_RAW

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Relanto Policy Assistant API", version="2.0")

# Instantiate global semantic cache
global_cache = SemanticRetrievalCache()

import asyncio

# Warmed up global cache pre-seeded at startup asynchronously in the background
async def preseed_cache_task():
    logger.info("Initializing global semantic cache and pre-seeding core FAQs in the background...")
    faqs = [
        # Core UI suggestion grid queries
        "What is the vacation policy for senior associates?",
        "How do I submit an expense report for travel?",
        "What are our data handling protocols for client PII?",
        "Can I request a standing desk through HR?",
        # Other standard queries
        "What are the rules for working from home?",
        "What is the POSH policy?",
        "What is the menstrual leave policy in Pune?",
        "What does the medical insurance cover?",
        "What is the appraisal and objective setting process?",
        "What are the official company holiday list and leave policies?"
    ]
    for faq in faqs:
        try:
            logger.info("Background pre-seeding FAQ: '%s'", faq)
            # Run the synchronous RAG pipeline inside a thread pool to avoid blocking the main event loop
            await asyncio.to_thread(
                retrieve_context,
                query=faq,
                top_k=3,
                semantic_cache=global_cache
            )
        except Exception as e:
            logger.error("Failed to pre-seed FAQ '%s' in background: %s", faq, e)
    logger.info("Background pre-seeding completed successfully. Cache is fully warmed!")

@app.on_event("startup")
async def startup_event():
    # Fire and forget the background pre-seeding task so the server opens its port instantly!
    asyncio.create_task(preseed_cache_task())


def simplify_source(filename: str, section: str) -> tuple[str, str, str]:
    """Map raw filenames and verbose sections to clean, human-friendly titles and doc codes."""
    mapping = {
        "Relanto Employee Handbook Jan 24 _ V 1.2_2025.pdf": ("Global Employee Handbook 2025", "DOC-2025-HB-01"),
        "POSH.pdf": ("Prevention of Sexual Harassment Policy", "DOC-2024-SH-02"),
        "Menstrual.pdf": ("Menstrual Leave Policy", "DOC-2024-ML-03"),
        "Relanto- Work From Home (WFH) Policy_Mar 26.pdf": ("Work From Home (WFH) Policy", "DOC-2024-WFH-04"),
        "Relanto - Insurance Policy_24-25.pdf": ("Group Medical Insurance Policy", "DOC-2024-INS-05"),
        "Relanto - PMS_2024_objective settings.pdf": ("Performance Management System (PMS)", "DOC-2024-PMS-06"),
        "PMS.pdf": ("Performance Management System (PMS)", "DOC-2024-PMS-06"),
        "PMS H2.pdf": ("Performance Management System (PMS)", "DOC-2024-PMS-06"),
        "ABC of AI Handbook_C1.pdf": ("AI Governance Handbook", "DOC-2024-AI-07"),
        "Data Security.pdf": ("Data Security Policy", "DOC-2024-SEC-08"),
    }
    
    clean_title, doc_id = mapping.get(filename, (filename.replace(".pdf", ""), "DOC-2024-GEN-09"))
    
    clean_section = section or "General Guideline"
    # Clean up extremely long raw sections
    if "Floater Leave 02" in clean_section:
        clean_section = "Section 7.0: Floater Leave"
    elif "The Complainant has the option" in clean_section:
        clean_section = "Section 1.0: Complaint Resolution"
        
    if len(clean_section) > 40:
        clean_section = clean_section[:37] + "..."
        
    return clean_title, doc_id, clean_section


class QueryRequest(BaseModel):
    query: str

class SourceItem(BaseModel):
    source_file: str
    raw_filename: str
    doc_id: str
    page: int
    section_title: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]
    cache_hit: bool
    cache_type: str



# Custom Friendly Prompt
USER_QA_SYSTEM_PROMPT = """You are a warm, extremely friendly employee concierge at Relanto.
You answer user queries about company policies using ONLY the provided contexts.

Rules:
1. First sentence MUST be a clean, direct, simplified summary calculation or statement highlighted in bold.
   - Example style: "You are entitled to 6 casual leaves per year. If you have used 3, you have exactly 3 days remaining."
   - Never output technical jargon, system logs, or JSON.
2. Underneath, write a simple 2-3 sentence friendly justification explaining the exact policy criteria from the text.
3. Keep the language natural, human, encouraging, and clear.
"""

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        # Run retrieval pipeline (each query is independent, no memory)
        retrieval = retrieve_context(
            query=query,
            top_k=3,
            semantic_cache=global_cache,
        )
        
        is_allowed = retrieval.get("allowed", True)
        if not is_allowed:
            return QueryResponse(
                query=query,
                answer="This query contains terms restricted by our policy guidelines. Please ask HR for details.",
                sources=[],
                cache_hit=False,
                cache_type="blocked"
            )
            
        parent_contexts = retrieval["parent_contexts"]
        answer = generate_answer(
            query=retrieval.get("effective_query") or query,
            contexts=parent_contexts,
            model="llama-3.3-70b-versatile",
            system_prompt=USER_QA_SYSTEM_PROMPT,
        )
        
        # Format deduplicated sources list
        sources_list = []
        seen = set()
        for ctx in parent_contexts:
            raw_src = ctx.get("source_file") or ctx.get("policy_name") or "Unknown"
            page = int(ctx.get("page", 0))
            raw_sec = ctx.get("section_title") or "General"
            
            clean_title, doc_id, clean_sec = simplify_source(raw_src, raw_sec)
            
            # Use tuple to deduplicate
            sig = (clean_title, page, clean_sec)
            if sig not in seen:
                seen.add(sig)
                sources_list.append(SourceItem(
                    source_file=clean_title,
                    raw_filename=raw_src,
                    doc_id=doc_id,
                    page=page,
                    section_title=clean_sec
                ))

                
        return QueryResponse(
            query=query,
            answer=answer,
            sources=sources_list,
            cache_hit=retrieval.get("cache_hit", False),
            cache_type=retrieval.get("cache_match_type", "miss")
        )
        
    except Exception as e:
        logger.error("Error in query endpoint: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def list_documents():
    raw_dir = Path(DATA_RAW)
    if not raw_dir.exists():
        return {"documents": []}
    files = sorted([f.name for f in raw_dir.glob("*.pdf")])
    return {"documents": files}

@app.get("/api/document/{filename}")
async def serve_document(filename: str):
    raw_path = Path(DATA_RAW) / filename
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(raw_path, media_type="application/pdf", filename=filename)

# Serve Frontend Root
@app.get("/")
async def get_index():
    index_path = PROJECT_ROOT / "frontend" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index_path)
