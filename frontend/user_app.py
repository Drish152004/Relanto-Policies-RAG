import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# Reuse the global semantic cache and retrieval functions
from frontend.app import get_global_semantic_cache
from llm.generate import generate_answer
from retrieval.memory import ConversationMemory
from retrieval.pipeline import retrieve_context
from utils.config import DATA_RAW

logger = logging.getLogger(__name__)


# Premium, minimalistic configuration
st.set_page_config(
    page_title="Relanto Employee Portal",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for pure premium elegance, Outfit font, glassmorphism, and simple visual cards
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Elegant Title Styling */
    .app-logo {
        text-align: center;
        margin-top: 1rem;
        font-size: 3rem;
    }
    
    .app-title {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    
    .app-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Clean Employee Focused Answer Box */
    .employee-answer-card {
        background: linear-gradient(135deg, rgba(167, 139, 250, 0.05) 0%, rgba(244, 114, 182, 0.02) 100%);
        border: 1px solid rgba(167, 139, 250, 0.15);
        border-radius: 18px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.1);
    }
    
    .direct-summary-badge {
        background: rgba(167, 139, 250, 0.1);
        border-left: 4px solid #a78bfa;
        color: #e9d5ff;
        padding: 1rem 1.2rem;
        border-radius: 4px 12px 12px 4px;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        line-height: 1.4;
    }
    
    .justification-content {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Interactive minimal buttons */
    .suggestion-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .source-tag {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .source-tag-name {
        font-weight: 500;
        font-size: 0.95rem;
        color: #cbd5e1;
    }
    
    .source-tag-meta {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render logo & titles
st.markdown("<div class='app-logo'>🌸</div>", unsafe_allow_html=True)
st.markdown("<div class='app-title'>Relanto Policy Concierge</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Your direct, friendly policy assistant. No jargon, just clear answers.</div>", unsafe_allow_html=True)

# Fetch warmed cache and set memory session states
global_cache = get_global_semantic_cache()
if "user_memory" not in st.session_state:
    st.session_state.user_memory = ConversationMemory()
if "user_query_input" not in st.session_state:
    st.session_state.user_query_input = ""

# Simple FAQs
USER_FAQS = [
    "What are the rules for working from home?",
    "What is the POSH policy?",
    "What is the menstrual leave policy in Pune?",
    "What does the medical insurance cover?",
    "What is the appraisal and objective setting process?",
    "What are the official company holiday list and leave policies?"
]

st.markdown("<div class='suggestion-title'>💡 Quick FAQs:</div>", unsafe_allow_html=True)
cols = st.columns(2)
for idx, faq in enumerate(USER_FAQS):
    col_idx = idx % 2
    if cols[col_idx].button(faq, key=f"user_faq_{idx}", use_container_width=True):
        st.session_state.user_query_input = faq
        st.rerun()

# Main Query input (automatically bound to st.session_state.user_query_input)
query = st.text_input(
    "Ask any policy question here:",
    key="user_query_input",
    placeholder="e.g., 'How many casual leaves do I have left?'",
)


# Friendly instructions override system prompt
USER_QA_SYSTEM_PROMPT = """You are a warm, extremely friendly employee concierge at Relanto.
You answer user queries about company policies using ONLY the provided contexts.

Rules:
1. First sentence MUST be a clean, direct, simplified summary calculation or statement highlighted in bold.
   - Example style: "You are entitled to 6 casual leaves per year. If you have used 3, you have exactly 3 days remaining."
   - Never output technical jargon, system logs, or JSON.
2. Underneath, write a simple 2-3 sentence friendly justification explaining the exact policy criteria from the text.
3. Keep the language natural, human, encouraging, and clear.
"""

if query:
    with st.spinner("Finding information for you..."):
        # Fetch contexts using high speed retrieval pipeline
        retrieval = retrieve_context(
            query=query,
            top_k=3,
            memory=st.session_state.user_memory,
            semantic_cache=global_cache,
        )
        
        is_allowed = retrieval.get("allowed", True)
        if not is_allowed:
            st.error("I'm sorry, but that query appears to contain terms that I cannot look up under our policy guidelines.")
        else:
            parent_contexts = retrieval["parent_contexts"]
            answer = generate_answer(
                query=retrieval.get("effective_query") or query,
                contexts=parent_contexts,
                model="llama-3.3-70b-versatile", # Use higher quality LLM for clean calculations
                system_prompt=USER_QA_SYSTEM_PROMPT,
            )
            
            # Format and display answer beautifully
            st.markdown("### 🌸 Here is what I found:")
            st.markdown(
                f"""
                <div class='employee-answer-card'>
                    <div class='direct-summary-badge'>{answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Display source files cleanly & add option to open the PDF directly
            if parent_contexts:
                st.markdown("<p style='font-size: 0.95rem; font-weight:600; color:#94a3b8; margin-top:2rem;'>📚 Official Source Reference:</p>", unsafe_allow_html=True)
                
                # Deduplicate sources
                seen_sources = {}
                for ctx in parent_contexts:
                    src = ctx.get("source_file") or ctx.get("policy_name")
                    page = ctx.get("page", "Unknown")
                    if src and src not in seen_sources:
                        seen_sources[src] = page
                
                for src_file, page in seen_sources.items():
                    # Create nice clean source layout
                    col_text, col_action = st.columns([3, 1])
                    
                    with col_text:
                        st.markdown(
                            f"""
                            <div class='source-tag'>
                                <span class='source-tag-name'>📄 {src_file}</span>
                                <span class='source-tag-meta'>Page: {page}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    
                    with col_action:
                        # Attempt to resolve local path for browser-safe download button
                        full_pdf_path = Path(DATA_RAW) / src_file
                        if os.path.exists(full_pdf_path):
                            try:
                                with open(full_pdf_path, "rb") as f:
                                    pdf_bytes = f.read()
                                st.download_button(
                                    label="📂 View PDF",
                                    data=pdf_bytes,
                                    file_name=src_file,
                                    mime="application/pdf",
                                    key=f"dl_{src_file}",
                                    use_container_width=True,
                                    help=f"Open and preview {src_file} directly"
                                )
                            except Exception as e:
                                st.error(f"Error loading file: {e}")
                        else:
                            st.button("📄 Not Found", key=f"not_found_{src_file}", disabled=True, use_container_width=True)

