import streamlit as st
from src.core.langchain_quanta import LangChainQuantaBot
import uuid
import atexit
import shutil
import tempfile
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check for OpenAI API key
def check_openai_key():
    """Check if OpenAI API key is available"""
    api_key = None

    # Try to get from Streamlit secrets first (for cloud deployment)
    try:
        api_key = st.secrets["openai"]["api_key"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitAPIException):
        # Try environment variable
        api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    return False

def check_authentication():
    """Check if user is authenticated with correct password"""
    correct_password = None

    try:
        correct_password = st.secrets["auth"]["password"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitAPIException):
        correct_password = os.getenv("APP_PASSWORD", "dongeunkim")

    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # If not authenticated, show login form
    if not st.session_state.authenticated:
        st.title("QuantaBot User Authentication")
        st.markdown("---")

        with st.container():
            st.subheader("Please enter the access password:")

            # Password input
            password_input = st.text_input(
                "Password",
                type="password",
                placeholder="Enter a valid password provided in advance."
            )

            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                login_button = st.button("Login", type="primary", use_container_width=True)

            if login_button:
                if password_input == correct_password:
                    st.session_state.authenticated = True
                    st.success("✅ Access granted. Welcome to QuantaBot!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Please try again.")

            st.markdown("---")
            st.info("Contact **dongeunk@umich.edu** for more information about this product.")

        return False

    return True

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
    # USER AUTHENTICATION
    if not check_authentication():
        st.stop()

    st.title("QuantaBot: Research-Aware Assistant")

    # Check for OpenAI API key
    if not check_openai_key():
        st.error("🔑 OpenAI API key not found!")
        st.stop()

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
        "Upload PDF/TXT/DOCX Documents",
        type=["pdf", "txt", "docx"],
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
                with st.spinner("Processing with LangChain integrated RAG pipeline..."):
                    start_time = time.time()

                    # Initialize LangChain QuantaBot
                    langchain_quanta = LangChainQuantaBot(
                        collection_name=st.session_state["collection_name"]
                    )

                    # Convert uploaded files to temp files while preserving original names
                    temp_file_paths = []
                    file_mapping = {}  # Map temp paths to original names

                    for f in st.session_state["files"]:
                        suffix = os.path.splitext(f.name)[1]
                        f.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp.flush()
                            temp_file_paths.append(tmp.name)
                            file_mapping[tmp.name] = f.name  # Store original filename

                    # Store file mapping for later use in citations
                    st.session_state["file_mapping"] = file_mapping

                    # Process documents
                    documents = langchain_quanta.preprocess_documents(temp_file_paths, file_mapping=file_mapping)

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
                    result = langchain_quanta.query(user_query)
                    answer = result["answer"]
                    source_documents = result["source_documents"]

                    response_time = time.time() - start_time

                    # Get retrieval stats
                    retrieval_stats = langchain_quanta.get_retrieval_stats()

                # Save to chat history
                st.session_state["chat_history"].append({
                    "question": user_query,
                    "answer": answer,
                    "source_documents": source_documents,
                    "response_time": response_time,
                    "retrieval_stats": retrieval_stats
                })

                # Display answer
                st.markdown("### 📝 Answer:")
                st.write(answer)

                # Display source citations
                if source_documents:
                    with st.expander("📚 Source Citations", expanded=True):
                        st.markdown("**Sources used in this response:**")

                        # Group documents by filename, filtering out unknown sources
                        sources_by_file = {}
                        for doc in source_documents:
                            filename = doc.metadata.get("filename", "Unknown")
                            page = doc.metadata.get("page", "N/A")

                            # Skip documents with empty metadata or unknown filenames
                            if not doc.metadata or filename == "Unknown":
                                continue

                            if filename not in sources_by_file:
                                sources_by_file[filename] = []

                            sources_by_file[filename].append({
                                "page": page,
                                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                            })

                        # Display sources grouped by file
                        if sources_by_file:
                            for filename, sources in sources_by_file.items():
                                st.markdown(f"**📄 {filename}**")

                                for source in sources:
                                    if source["page"] != "N/A":
                                        st.markdown(f"- **Page {source['page']}:** {source['content']}")
                                    else:
                                        st.markdown(f"- {source['content']}")

                                st.markdown("---")
                        else:
                            st.markdown("*No identifiable source citations available for this response.*")

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
        st.header("Conversation History")

        for i, entry in enumerate(reversed(st.session_state["chat_history"])):
            with st.expander(f"Q{len(st.session_state['chat_history'])-i}: {entry['question'][:100]}...", expanded=(i==0)):
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**Answer:** {entry['answer']}")

                # Show source citations if available
                if "source_documents" in entry and entry["source_documents"]:
                    st.markdown("**📚 Sources:**")
                    sources_by_file = {}
                    for doc in entry["source_documents"]:
                        filename = doc.metadata.get("filename", "Unknown")
                        page = doc.metadata.get("page", "N/A")

                        if filename not in sources_by_file:
                            sources_by_file[filename] = []
                        sources_by_file[filename].append(page)

                    # Display compact source list
                    for filename, pages in sources_by_file.items():
                        if "N/A" not in pages:
                            pages_str = ", ".join([f"p.{p}" for p in sorted(set(pages)) if p != "N/A"])
                            st.caption(f"📄 {filename}: {pages_str}")
                        else:
                            st.caption(f"📄 {filename}")

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