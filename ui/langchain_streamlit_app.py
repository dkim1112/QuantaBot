import streamlit as st
from src.core.langchain_quanta import LangChainQuantaBot
import uuid
import atexit
import shutil
import tempfile
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Streamlit configuration
st.set_page_config(page_title="QuantaBot", layout="wide")
st.markdown("""
    <style>
    section.main > div {
        max-width: none !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .retrieval-stats {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def cleanup_chroma_folder(path):
    def _cleanup():
        shutil.rmtree(path, ignore_errors=True)
        print(f"🧹 Deleted Chroma folder: {path}")
    return _cleanup

def streamlit_ui():
    st.title("QuantaBot: Research-Aware Assistant")

    # Initialize session state for LangChain components
    if "files" not in st.session_state:
        st.session_state["files"] = None

    if "collection_name" not in st.session_state:
        st.session_state["collection_name"] = f"langchain_collection_{uuid.uuid4().hex[:8]}"

    if "langchain_quanta" not in st.session_state:
        st.session_state["langchain_quanta"] = None

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "processing_stats" not in st.session_state:
        st.session_state["processing_stats"] = {}

    # File upload section
    st.header("Document Upload & Processing")
    files = st.file_uploader(
        "Upload PDF/TXT Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="langchain_file_uploader"
    )

    if files:
        if files != st.session_state["files"]:
            st.session_state["files"] = files
            st.session_state["collection_name"] = f"langchain_collection_{uuid.uuid4().hex[:8]}"
            st.session_state["langchain_quanta"] = None
            st.session_state["chat_history"] = []
            st.session_state["processing_stats"] = {}

    # Document preprocessing
    if st.button("Process Document(s)", type="primary"):
        if st.session_state["files"] is None:
            st.warning("Please upload documents first.")
        else:
            try:
                with st.spinner("Processing with LangChain advanced RAG pipeline..."):
                    start_time = time.time()

                    # Initialize LangChain QuantaBot
                    langchain_quanta = LangChainQuantaBot(
                        collection_name=st.session_state["collection_name"]
                    )

                    # Convert uploaded files to temp files
                    temp_file_paths = []
                    for f in st.session_state["files"]:
                        suffix = os.path.splitext(f.name)[1]
                        f.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp.flush()
                            temp_file_paths.append(tmp.name)

                    # Process documents
                    documents = langchain_quanta.preprocess_documents(temp_file_paths)

                    processing_time = time.time() - start_time

                    # Store in session state
                    st.session_state["langchain_quanta"] = langchain_quanta
                    st.session_state["processing_stats"] = {
                        "documents_processed": len(documents),
                        "processing_time": processing_time,
                        "retriever_type": type(langchain_quanta.retriever).__name__
                    }

                    # Register cleanup
                    if hasattr(langchain_quanta, "persist_directory") and langchain_quanta.persist_directory:
                        atexit.register(cleanup_chroma_folder(langchain_quanta.persist_directory))

                    st.success(f"Ready! Processed {len(documents)} chunks in {processing_time:.1f}s")

            except Exception as e:
                st.error(f"❌ Error during processing: {e}")

    # Display processing stats
    if st.session_state["processing_stats"]:
        stats = st.session_state["processing_stats"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Processed", stats["documents_processed"])
        with col2:
            st.metric("Processing Time", f"{stats['processing_time']:.1f}s")
        with col3:
            st.metric("Retriever Type", stats["retriever_type"])

    # Query section
    st.header("Chat Interface")

    documents_uploaded = st.session_state.get("files") is not None
    documents_processed = st.session_state.get("langchain_quanta") is not None
    ready_for_questions = documents_uploaded and documents_processed

    if not ready_for_questions:
        st.info("Please upload and process documents to start chatting.")

    # Chat interface
    user_query = st.text_input(
        "Ask your question:",
        disabled=not ready_for_questions,
        placeholder="e.g., What are the main findings of this research?"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        ask_button = st.button("Ask", disabled=not ready_for_questions)

    with col2:
        clear_memory = st.button("Clear Conversation Memory", disabled=not ready_for_questions)

    if clear_memory and st.session_state.get("langchain_quanta"):
        st.session_state["langchain_quanta"].clear_memory()
        st.session_state["chat_history"] = []
        st.success("Conversation memory cleared!")

    # Process query
    if ask_button and user_query.strip():
        langchain_quanta = st.session_state.get("langchain_quanta")
        if langchain_quanta is None:
            st.error("Please process documents first.")
        else:
            try:
                with st.spinner("🤔 Thinking..."):
                    start_time = time.time()

                    # Get answer using LangChain RAG
                    answer = langchain_quanta.query(user_query)

                    response_time = time.time() - start_time

                    # Get retrieval stats
                    retrieval_stats = langchain_quanta.get_retrieval_stats()

                # Save to chat history
                st.session_state["chat_history"].append({
                    "question": user_query,
                    "answer": answer,
                    "response_time": response_time,
                    "retrieval_stats": retrieval_stats
                })

                # Display answer
                st.markdown("### Answer:")
                st.write(answer)

                # Show performance stats
                with st.expander("📊 Performance Statistics", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Response Time", f"{response_time:.2f}s")
                        st.metric("Total Queries", retrieval_stats.get("total_queries", 0))

                    with col2:
                        if retrieval_stats.get("retrieval_stats"):
                            last_retrieval = retrieval_stats["retrieval_stats"][-1]
                            st.metric("Documents Retrieved", last_retrieval.get("doc_count", 0))
                            st.metric("Avg Doc Length", f"{last_retrieval.get('avg_length', 0):.0f}")

            except Exception as e:
                st.error(f"❌ Error processing query: {e}")

    # Chat History
    if st.session_state["chat_history"]:
        st.markdown("---")
        st.header("💬 Conversation History")

        for i, entry in enumerate(reversed(st.session_state["chat_history"])):
            with st.expander(f"Q{len(st.session_state['chat_history'])-i}: {entry['question'][:100]}...", expanded=(i==0)):
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**Answer:** {entry['answer']}")

                # Show stats if available
                if "response_time" in entry:
                    st.caption(f"Response time: {entry['response_time']:.2f}s")

    # System info
    with st.sidebar:
        st.header("System Information")

        if st.session_state.get("langchain_quanta"):
            quanta = st.session_state["langchain_quanta"]
            st.write(f"**Collection:** {quanta.collection_name}")

            if hasattr(quanta, 'retriever') and quanta.retriever:
                st.write(f"**Retriever:** {type(quanta.retriever).__name__}")

            # Memory info
            if hasattr(quanta.memory, 'chat_memory'):
                msg_count = len(quanta.memory.chat_memory.messages)
                st.write(f"**Memory:** {msg_count} messages")

if __name__ == "__main__":
    streamlit_ui()