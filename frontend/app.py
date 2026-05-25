"""Streamlit RAG frontend: Chat interface, source cards, and query diagnostic logs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from llm.generate import generate_answer
from retrieval.cache import SemanticRetrievalCache
from retrieval.memory import ConversationMemory
from retrieval.pipeline import retrieve_context
from utils.config import DATA_RAW

# Set page config for a premium layout
st.set_page_config(
    page_title="Relanto Policy RAG",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich styling, Outfit typography, and glassmorphism cards
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global font styles */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main container styling */
    .main-title {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Translucent glassmorphism panels */
    .glass-panel {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
    }
    
    /* Answer box gradient styling */
    .answer-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.07) 0%, rgba(168, 85, 247, 0.04) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Interactive source cards */
    .source-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #818cf8;
        border-top: 1px solid rgba(255, 255, 255, 0.02);
        border-right: 1px solid rgba(255, 255, 255, 0.02);
        border-bottom: 1px solid rgba(255, 255, 255, 0.02);
        border-radius: 0px 12px 12px 0px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.06);
        border-left-color: #c084fc;
        box-shadow: 0 8px 24px rgba(129, 140, 248, 0.15);
    }
    
    /* Modern badges */
    .badge-primary {
        background: rgba(129, 140, 248, 0.15);
        color: #818cf8;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        display: inline-block;
        border: 1px solid rgba(129, 140, 248, 0.3);
    }
    
    .badge-secondary {
        background: rgba(192, 132, 252, 0.15);
        color: #c084fc;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        display: inline-block;
        border: 1px solid rgba(192, 132, 252, 0.3);
    }
    
    .source-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.4rem;
    }
    
    .source-snippet {
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    
    /* Diagnostics labels */
    .diag-label {
        font-weight: 600;
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    
    .diag-val {
        color: #f1f5f9;
        font-size: 1rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

import logging
logger = logging.getLogger(__name__)

# Sidebar layout
st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=64)
st.sidebar.markdown("### Configuration")

@st.cache_resource
def get_global_semantic_cache() -> SemanticRetrievalCache:
    cache = SemanticRetrievalCache()
    logger.info("Initializing global semantic cache and pre-seeding core FAQs...")
    
    faqs = [
        "What are the rules for working from home?",
        "What is the POSH policy?",
        "What is the menstrual leave policy in Pune?",
        "What does the medical insurance cover?",
        "What is the appraisal and objective setting process?",
        "What are the official company holiday list and leave policies?"
    ]
    
    for faq in faqs:
        try:
            logger.info("Pre-seeding semantic cache for FAQ: %s", faq)
            retrieve_context(
                query=faq,
                top_k=3,
                semantic_cache=cache
            )
        except Exception as e:
            logger.error("Failed to pre-seed FAQ '%s': %s", faq, e)
            
    return cache

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory()

global_cache = get_global_semantic_cache()


# Dynamic list of raw PDF files for filtering
available_files = []
if os.path.exists(DATA_RAW):
    available_files = sorted([f for f in os.listdir(DATA_RAW) if f.endswith(".pdf")])

selected_files = st.sidebar.multiselect(
    "Filter by Source Document",
    options=available_files,
    help="Restrict retrieval to selected policies. Leave blank for auto-routing.",
)

# Retrieval tuning
top_k = st.sidebar.slider(
    "Context Limit (top-k parents)",
    min_value=1,
    max_value=10,
    value=3,
    help="Number of final parent context paragraphs to feed the generator model.",
)

# LLM model options
model_option = st.sidebar.selectbox(
    "Generation Model",
    options=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    index=0,
)

# Main page header
st.markdown("<div class='main-title'>Relanto Policy Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Instant, grounded answers across all company policies and employee guidelines</div>", unsafe_allow_html=True)

# Session state to hold selected FAQ query
if "selected_faq" not in st.session_state:
    st.session_state.selected_faq = ""

# Core FAQ suggestions matching document types
FAQ_SUGGESTIONS = [
    "What are the rules for working from home?",
    "What is the POSH policy?",
    "What is the menstrual leave policy in Pune?",
    "What does the medical insurance cover?",
    "What is the appraisal and objective setting process?",
    "What are the official company holiday list and leave policies?"
]

st.markdown("<p style='font-size: 0.95rem; color: #94a3b8; font-weight: 500; margin-bottom: 0.3rem;'>💡 Warmed Policy FAQ Shortcuts (Instant Answers):</p>", unsafe_allow_html=True)
faq_cols = st.columns(3)
for idx, faq in enumerate(FAQ_SUGGESTIONS):
    col_idx = idx % 3
    if faq_cols[col_idx].button(faq, key=f"faq_btn_{idx}", use_container_width=True):
        st.session_state.selected_faq = faq
        st.rerun()

# Determine default query
default_query = ""
if st.session_state.selected_faq:
    default_query = st.session_state.selected_faq
    st.session_state.selected_faq = ""

# Main Query input
query = st.text_input(
    "Search policies (e.g., 'What is the menstrual leave policy in Pune?' or 'What are the remote work rules?'):",
    value=default_query,
    key="query_input",
    placeholder="Ask something...",
)

if query:
    with st.spinner("Processing query and retrieving policy context..."):
        # 1. Trigger the retrieval pipeline
        force_filters = selected_files if selected_files else None
        retrieval_results = retrieve_context(
            query=query,
            top_k=top_k,
            force_source_files=force_filters,
            memory=st.session_state.conversation_memory,
            semantic_cache=global_cache,
        )


        is_allowed = retrieval_results.get("allowed", True)
        if is_allowed:
            # 2. Trigger answer generation
            parent_contexts = retrieval_results["parent_contexts"]
            answer = generate_answer(
                query=retrieval_results.get("effective_query") or query,
                contexts=parent_contexts,
                model=model_option,
            )
        else:
            parent_contexts = []
            answer = None

    # Handle display based on security validation
    if not is_allowed:
        st.markdown("### 🚫 Blocked by Guardrails")
        st.error(
            f"**Query Blocked by AI Security Guardrails**\n\n"
            f"- **Risk Type:** `{retrieval_results.get('risk_type')}`\n"
            f"- **Reason:** {retrieval_results.get('reason')}"
        )
    else:
        # Display Answer Panel
        st.markdown("### 💬 Grounded Response")
        st.markdown(
            f"<div class='answer-box'>{answer}</div>",
            unsafe_allow_html=True,
        )

    # Split display for Diagnostics and Sources
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🔎 Query Analysis & Diagnostics")
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        
        st.markdown("<div class='diag-label'>Guardrail Status</div>", unsafe_allow_html=True)
        status_color = "#2cb67d" if is_allowed else "#ef4444"
        status_text = "PASSED" if is_allowed else "BLOCKED"
        st.markdown(f"<div class='diag-val'><b style='color: {status_color};'>{status_text}</b></div>", unsafe_allow_html=True)
        
        if is_allowed:
            st.markdown("<div class='diag-label'>Optimized Retrieval Query</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='diag-val'><i>\"{retrieval_results['optimized_query']}\"</i></div>", unsafe_allow_html=True)

            memory_context = retrieval_results.get("memory_context") or {}
            if memory_context.get("resolved_from_memory"):
                st.markdown("<div class='diag-label'>Conversation Memory</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='diag-val'>Follow-up resolved against: {memory_context.get('previous_topic', 'previous policy topic')}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='diag-label'>Retrieval Cache</div>", unsafe_allow_html=True)
            cache_status = "HIT" if retrieval_results.get("cache_hit") else "MISS"
            cache_conf = float(retrieval_results.get("cache_confidence") or 0.0)
            cache_type = retrieval_results.get("cache_match_type") or "miss"
            st.markdown(
                f"<div class='diag-val'><b>{cache_status}</b> ({cache_type}, confidence {cache_conf:.3f})</div>",
                unsafe_allow_html=True,
            )
            
            st.markdown("<div class='diag-label'>Extracted Technical Keywords</div>", unsafe_allow_html=True)
            keywords_html = " ".join([f"<span class='badge-primary'>{kw}</span>" for kw in retrieval_results["keywords"]])
            if not keywords_html:
                keywords_html = "<span style='color: #64748b;'>None</span>"
            st.markdown(f"<div style='margin-bottom: 0.8rem;'>{keywords_html}</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='diag-label'>Semantic Intent</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='diag-val'>{retrieval_results['semantic_intent']}</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='diag-label'>Document Source Routing</div>", unsafe_allow_html=True)
            routed_docs = retrieval_results["source_files"]
            docs_html = " ".join([f"<span class='badge-secondary'>{doc}</span>" for doc in routed_docs])
            if not docs_html:
                docs_html = "<span class='badge-secondary'>All Documents (Auto)</span>"
            st.markdown(f"<div>{docs_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='diag-label'>Blocked Query</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='diag-val'><i>\"{query}\"</i></div>", unsafe_allow_html=True)
            st.markdown("<div class='diag-label'>Risk Type</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='diag-val'><code>{retrieval_results.get('risk_type')}</code></div>", unsafe_allow_html=True)
            st.markdown("<div class='diag-label'>Reason</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='diag-val'>{retrieval_results.get('reason')}</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 📚 Grounding Excerpts")
        if not is_allowed:
            st.warning("Retrieval skipped due to guardrail block.")
        elif not parent_contexts:
            st.info("No matching policy text found in database.")
        else:
            for idx, ctx in enumerate(parent_contexts):
                doc_name = ctx.get("source_file") or ctx.get("policy_name") or "Policy Document"
                section_title = ctx.get("section_title") or "General"
                page = ctx.get("page", "Unknown")
                text = ctx.get("parent_text", "")
                score = ctx.get("score", 0.0)

                st.markdown(
                    f"""
                    <div class='source-card'>
                        <div class='source-title'>[{idx + 1}] {doc_name}</div>
                        <div style='margin-bottom: 0.6rem;'>
                            <span class='badge-primary'>Page {page}</span>
                            <span class='badge-secondary'>{section_title}</span>
                            <span class='badge-primary' style='background: rgba(44, 182, 125, 0.15); color: #2cb67d; border-color: rgba(44, 182, 125, 0.3);'>Relevance: {score:.3f}</span>
                        </div>
                        <div class='source-snippet'>{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
else:
    # Display welcoming guidelines when no query is typed
    st.info("💡 Type a question above to retrieve policy details and generate grounded responses.")
    
    st.markdown("### 📂 Available Policies")
    if available_files:
        cols = st.columns(3)
        for i, file in enumerate(available_files):
            col_idx = i % 3
            cols[col_idx].markdown(
                f"""
                <div class='glass-panel' style='padding: 1rem; text-align: center;'>
                    📄 <b style='font-size: 0.95rem; color:#fff;'>{file}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("No PDF policies found in raw data folder.")

