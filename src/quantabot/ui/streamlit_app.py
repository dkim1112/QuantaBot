import streamlit as st

if "page_config_set" not in st.session_state:
    st.set_page_config(page_title="QuantaBot", layout="wide")
    st.session_state["page_config_set"] = True

import atexit
import os
import shutil
import tempfile
import time
import uuid

from quantabot.core.rag import LangChainQuantaBot

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@st.cache_resource
def load_embedding_model():
    from quantabot.utils.embedding_wrapper import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings()


@st.cache_resource
def load_cross_encoder_model():
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    return HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


def check_openai_key() -> bool:
    api_key = None
    try:
        api_key = st.secrets["openai"]["api_key"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitAPIException):
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    return False


def check_authentication() -> bool:
    """Gate the app behind a password. Fails closed when no password is configured."""
    correct_password = None
    try:
        correct_password = st.secrets["auth"]["password"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitAPIException):
        correct_password = os.getenv("APP_PASSWORD")

    if not correct_password:
        st.error(
            "🔒 No access password is configured. "
            "Set `APP_PASSWORD` in the environment or `[auth] password` in `.streamlit/secrets.toml`."
        )
        return False

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("QuantaBot — Access")
    st.markdown("---")
    with st.container():
        password_input = st.text_input(
            "Password", type="password", placeholder="Enter access password"
        )
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Login", type="primary", use_container_width=True):
                if password_input == correct_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password.")
        st.markdown("---")
        st.info("Contact **dongeunk@umich.edu** for access.")
    return False


st.markdown(
    """
    <style>
    section.main > div {
        max-width: none !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def cleanup_chroma_folder(path):
    def _cleanup():
        shutil.rmtree(path, ignore_errors=True)
    return _cleanup


def _files_signature(uploaded_files) -> str:
    """Stable signature for the current upload set, used to detect changes."""
    return "|".join(f"{f.name}:{f.size}" for f in uploaded_files)


def _process_uploaded_files(uploaded_files):
    """Embed and index the uploaded files into a fresh QuantaBot instance."""
    embedding_model = load_embedding_model()
    collection_name = f"langchain_collection_{uuid.uuid4().hex[:8]}"
    quanta = LangChainQuantaBot(
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    temp_paths = []
    file_mapping = {}
    for f in uploaded_files:
        suffix = os.path.splitext(f.name)[1]
        f.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            tmp.flush()
            temp_paths.append(tmp.name)
            file_mapping[tmp.name] = f.name

    start = time.time()
    documents = quanta.preprocess_documents(temp_paths, file_mapping=file_mapping)
    elapsed = time.time() - start

    if getattr(quanta, "persist_directory", None):
        atexit.register(cleanup_chroma_folder(quanta.persist_directory))

    return quanta, {
        "documents_processed": len(documents),
        "processing_time": elapsed,
        "retriever_type": type(quanta.retriever).__name__,
        "file_mapping": file_mapping,
    }


def _render_citations(source_documents):
    """Render full retrieved chunks grouped by file with a chunk index."""
    if not source_documents:
        return
    grouped = {}
    for idx, doc in enumerate(source_documents, 1):
        metadata = getattr(doc, "metadata", {}) or {}
        filename = metadata.get("filename", "Unknown")
        if filename == "Unknown":
            continue
        grouped.setdefault(filename, []).append(
            {
                "idx": idx,
                "page": metadata.get("page", "N/A"),
                "content": doc.page_content,
            }
        )
    if not grouped:
        return
    with st.expander(f"📚 Sources ({len(source_documents)} chunks)", expanded=False):
        for filename, chunks in grouped.items():
            st.markdown(f"**📄 {filename}**")
            for chunk in chunks:
                page_label = (
                    f"chunk {chunk['idx']}, page {chunk['page']}"
                    if chunk["page"] != "N/A"
                    else f"chunk {chunk['idx']}"
                )
                with st.expander(page_label, expanded=False):
                    st.markdown(chunk["content"])
            st.markdown("---")


def streamlit_ui():
    if not check_authentication():
        st.stop()

    st.title("QuantaBot: Research-Aware Assistant")

    if not check_openai_key():
        st.error("🔑 OpenAI API key not found! Set `OPENAI_API_KEY` or `[openai] api_key` in secrets.")
        st.stop()

    # Session state
    st.session_state.setdefault("files_signature", None)
    st.session_state.setdefault("quanta", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("processing_stats", {})

    # Sidebar
    with st.sidebar:
        st.header("System")
        quanta = st.session_state.get("quanta")
        if quanta is not None:
            st.write(f"**Collection:** `{quanta.collection_name}`")
            if getattr(quanta, "retriever", None):
                st.write(f"**Retriever:** {type(quanta.retriever).__name__}")
            if hasattr(quanta.memory, "chat_memory"):
                st.write(f"**Memory:** {len(quanta.memory.chat_memory.messages)} messages")
        if st.button("Clear conversation", disabled=quanta is None, use_container_width=True):
            if quanta is not None:
                quanta.clear_memory()
            st.session_state["chat_history"] = []
            st.rerun()

    # Upload + auto-process
    st.header("Document upload")
    uploaded = st.file_uploader(
        "Upload PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded:
        sig = _files_signature(uploaded)
        if sig != st.session_state["files_signature"]:
            st.session_state["files_signature"] = sig
            st.session_state["quanta"] = None
            st.session_state["chat_history"] = []
            st.session_state["processing_stats"] = {}
            try:
                with st.status("Embedding and indexing documents…", expanded=False) as status:
                    quanta, stats = _process_uploaded_files(uploaded)
                    status.update(
                        label=f"Indexed {stats['documents_processed']} chunks in {stats['processing_time']:.1f}s",
                        state="complete",
                    )
                st.session_state["quanta"] = quanta
                st.session_state["processing_stats"] = stats
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")

    if st.session_state["processing_stats"]:
        s = st.session_state["processing_stats"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Chunks", s["documents_processed"])
        col2.metric("Time", f"{s['processing_time']:.1f}s")
        col3.metric("Retriever", s["retriever_type"])

    # Chat
    st.header("Chat")
    ready = st.session_state.get("quanta") is not None
    if not ready:
        st.info("Upload documents above to start chatting.")

    # Replay history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_citations(msg.get("source_documents", []))
                rt = msg.get("response_time")
                if rt is not None:
                    st.caption(f"⏱ {rt:.2f}s")

    prompt = st.chat_input(
        "Ask your question…",
        disabled=not ready,
    )

    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            quanta = st.session_state["quanta"]
            placeholder = st.empty()
            start = time.time()
            try:
                streamed = placeholder.write_stream(quanta.query_stream(prompt))
                response_time = time.time() - start
                full_answer = streamed if isinstance(streamed, str) else getattr(quanta, "last_answer", "")
                sources = getattr(quanta, "last_sources", [])
                _render_citations(sources)
                st.caption(f"⏱ {response_time:.2f}s")
                st.session_state["chat_history"].append(
                    {
                        "role": "assistant",
                        "content": full_answer,
                        "source_documents": sources,
                        "response_time": response_time,
                    }
                )
            except Exception as e:
                placeholder.error(f"❌ Error processing query: {e}")


if __name__ == "__main__":
    streamlit_ui()
