from __future__ import annotations

from pathlib import Path

import streamlit as st

from memory_rag_utils import ConversationMemory, chunks_from_text, load_docs, run_turn


ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS = ROOT / "sample_data" / "framework_docs.csv"


def read_uploaded_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".txt", ".md"}:
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception:
            return "PDF support requires pypdf to be installed."

        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    return "Unsupported file type. Upload a PDF, TXT, or MD file."


st.set_page_config(page_title="Day 11 Document Chat", page_icon="💬", layout="wide")
st.title("Day 11: Document Chat Assistant")
st.caption("History-aware retrieval with query reformulation and clear-history control.")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=4)
if "docs" not in st.session_state:
    st.session_state.docs = load_docs(DEFAULT_DOCS)
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

uploaded = st.file_uploader("Upload a PDF, TXT, or Markdown file", type=["pdf", "txt", "md"])
if uploaded is not None:
    uploaded_text = read_uploaded_text(uploaded)
    if uploaded_text.startswith("Unsupported") or uploaded_text.startswith("PDF support"):
        st.warning(uploaded_text)
    else:
        st.session_state.docs = chunks_from_text(uploaded_text)
        st.success(f"Indexed {len(st.session_state.docs)} chunks from {uploaded.name}.")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Clear History"):
        st.session_state.memory.clear()
        st.session_state.chat_log = []
        st.success("Conversation history cleared.")

query = st.chat_input("Ask a question about the document")
if query:
    result = run_turn(query, st.session_state.memory, st.session_state.docs)
    st.session_state.chat_log.append(("user", query))
    st.session_state.chat_log.append(("assistant", result["answer"]))
    st.session_state.last_result = result

for role, message in st.session_state.chat_log:
    with st.chat_message(role):
        st.write(message)

if "last_result" in st.session_state:
    st.subheader("Last Retrieval")
    st.write(f"Standalone query: `{st.session_state.last_result['standalone_query']}`")
    for source in st.session_state.last_result["sources"]:
        st.markdown(f"- **{source.title}** (`{source.doc_id}`): {source.text}")
