import sys
import os
import importlib.util
import importlib.abc

class _DirectFileFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        mod_name = fullname.split('.')[-1]
        search_paths = path or sys.path
        for p in search_paths:
            if not isinstance(p, str): continue
            candidate = os.path.join(p, f'{mod_name}.py')
            if os.path.isfile(candidate):
                return importlib.util.spec_from_file_location(fullname, candidate)
            pkg_init = os.path.join(p, mod_name, '__init__.py')
            if os.path.isfile(pkg_init):
                return importlib.util.spec_from_file_location(fullname, pkg_init, submodule_search_locations=[os.path.join(p, mod_name)])
        return None

sys.meta_path.insert(0, _DirectFileFinder())

import os
import sys
import time
import csv
from datetime import datetime
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from rag_pipeline import RAGPipeline
from prompt import DISCLAIMER_MESSAGE

FEEDBACK_LOG_PATH = os.path.join(current_dir, "user_feedback.csv")

st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F3F4F6;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        border: 1px solid #E5E7EB;
    }
    .feedback-status {
        font-size: 0.85rem;
        color: #059669;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


def log_feedback(question: str, answer: str, feedback_type: str, sources: list):
    file_exists = os.path.isfile(FEEDBACK_LOG_PATH)
    try:
        with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "feedback", "question", "answer", "source_count"])
            writer.writerow([
                datetime.now().isoformat(),
                feedback_type,
                question.replace("\n", " "),
                answer.replace("\n", " "),
                len(sources)
            ])
    except Exception as e:
        st.warning(f"Could not log feedback: {e}")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

if "index_loaded" not in st.session_state:
    saved = st.session_state.pipeline.load_existing_index()
    st.session_state.index_loaded = saved

if "feedback_records" not in st.session_state:
    st.session_state.feedback_records = {}


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/books.png", width=64)
    st.title("Document Manager")
    st.caption("PDF Upload & Ingestion")

    uploaded_files = st.file_uploader(
        "Upload Domain PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload 1 to 5 PDF documents (e.g., Company Policies, Employee Handbook). Maximum 15 MB per file."
    )

    if uploaded_files:
        st.write(f"**Selected Documents ({len(uploaded_files)}):**")
        for uf in uploaded_files:
            file_mb = len(uf.getvalue()) / (1024 * 1024)
            st.markdown(f"- 📄 `{uf.name}` *({file_mb:.2f} MB)*")

    col_proc, col_clr = st.columns(2)
    with col_proc:
        process_btn = st.button("🚀 Process", type="primary", use_container_width=True)
    with col_clr:
        clear_docs_btn = st.button("🔄 Clear", use_container_width=True)

    if clear_docs_btn:
        st.session_state.indexed_docs = []
        st.session_state.index_loaded = False
        st.session_state.pipeline = RAGPipeline()
        st.success("Documents cleared. You can upload new files.")
        st.rerun()

    if process_btn:
        if not uploaded_files:
            st.error("Please upload at least one valid PDF document.")
        else:
            with st.spinner("Extracting text, chunking, and generating FAISS vector embeddings..."):
                file_tuples = [(uf.getvalue(), uf.name) for uf in uploaded_files]
                result = st.session_state.pipeline.ingest_documents(file_tuples)

                if result["success"]:
                    st.session_state.index_loaded = True
                    st.session_state.indexed_docs = [uf.name for uf in uploaded_files]
                    st.success(f"Indexed {result['chunks_count']} chunks across {result['pages_count']} pages!")
                    st.rerun()
                else:
                    st.error(f"Processing failed: {result.get('error')}")
                    for err in result.get("validation_errors", []):
                        st.warning(err)

    st.divider()
    st.subheader("System Status")
    if st.session_state.index_loaded:
        st.success("🟢 Vector Index Ready")
        if st.session_state.indexed_docs:
            st.caption(f"Active Docs: {', '.join(st.session_state.indexed_docs)}")
        else:
            st.caption("Active Docs: Loaded from local saved_index")
    else:
        st.warning("🟡 No Index Loaded. Please upload and process documents.")

    st.divider()
    st.subheader("⚙️ Model Configuration")
    provider_choice = st.selectbox(
        "LLM Provider",
        options=["Auto / Local Grounded", "Groq (Llama-3.1-70B)", "OpenAI (GPT-4o-mini)", "Google Gemini"],
        index=0
    )
    api_key_input = st.text_input("API Key (Optional)", type="password", help="Enter API key for Groq, OpenAI, or Gemini. If omitted, the deterministic grounded extractor runs offline.")

    provider_map = {
        "Auto / Local Grounded": "auto",
        "Groq (Llama-3.1-70B)": "groq",
        "OpenAI (GPT-4o-mini)": "openai",
        "Google Gemini": "gemini"
    }
    st.session_state.pipeline.llm_provider = provider_map[provider_choice]
    st.session_state.pipeline.api_key = api_key_input if api_key_input.strip() else None

    st.divider()
    if st.button("📁 Load Default Sample Documents"):
        sample_dir = os.path.join(current_dir, "documents")
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith(".pdf")]
            file_tuples = []
            for f in sample_files:
                p = os.path.join(sample_dir, f)
                with open(p, "rb") as fp:
                    file_tuples.append((fp.read(), f))
            with st.spinner("Ingesting sample documents..."):
                res = st.session_state.pipeline.ingest_documents(file_tuples)
                if res["success"]:
                    st.session_state.index_loaded = True
                    st.session_state.indexed_docs = sample_files
                    st.success(f"Loaded {len(sample_files)} sample documents!")
                    st.rerun()


st.markdown('<div class="main-title">Domain-Specific RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent document question answering with verified page-level source attribution and strict guardrails.</div>', unsafe_allow_html=True)

st.markdown(f'<div class="disclaimer-box">{DISCLAIMER_MESSAGE}</div>', unsafe_allow_html=True)

col_status, col_clear_chat = st.columns([4, 1])
with col_status:
    st.caption("Ask questions strictly answered by the context of your uploaded documents.")
with col_clear_chat:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📄 View Sources ({len(msg['sources'])} retrieved citations)", expanded=False):
                for s_idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {s_idx}:** `{src['document']}` | **Page {src['page']}**")
                    st.markdown(f'<div class="source-box">"...{src["snippet"]}..."</div>', unsafe_allow_html=True)
                    st.markdown("")

        if msg["role"] == "assistant":
            msg_key = f"msg_{idx}"
            current_feedback = st.session_state.feedback_records.get(msg_key)

            if current_feedback:
                st.markdown(f'<span class="feedback-status">Feedback recorded: {current_feedback}</span>', unsafe_allow_html=True)
            else:
                f_col1, f_col2, f_col3 = st.columns([1, 1, 8])
                with f_col1:
                    if st.button("👍 Useful", key=f"thumb_up_{idx}"):
                        st.session_state.feedback_records[msg_key] = "Useful"
                        log_feedback(
                            question=st.session_state.messages[idx - 1]["content"] if idx > 0 else "Unknown",
                            answer=msg["content"],
                            feedback_type="Useful",
                            sources=msg.get("sources", [])
                        )
                        st.rerun()
                with f_col2:
                    if st.button("👎 Incorrect", key=f"thumb_down_{idx}"):
                        st.session_state.feedback_records[msg_key] = "Incorrect"
                        log_feedback(
                            question=st.session_state.messages[idx - 1]["content"] if idx > 0 else "Unknown",
                            answer=msg["content"],
                            feedback_type="Incorrect",
                            sources=msg.get("sources", [])
                        )
                        st.rerun()


user_query = st.chat_input("Ask a question about your uploaded documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    if not st.session_state.index_loaded:
        bot_response = "⚠️ No documents have been processed yet. Please upload and process PDF documents using the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": bot_response, "sources": []})
        st.rerun()

    with st.spinner("Searching document embeddings and generating grounded answer..."):
        response = st.session_state.pipeline.answer_question(user_query)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "refusal": response.get("refusal", False)
        })

    st.rerun()
