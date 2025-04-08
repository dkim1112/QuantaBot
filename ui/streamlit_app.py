import streamlit as st
from src.core.quanta import Quanta
import uuid
import atexit
import shutil
import tempfile
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# this needs to stay on top of any other file! (streamlit rule)
st.set_page_config(page_title="Quanta Chatbot", layout="wide")
st.markdown("""
    <style>
    section.main > div {
        max-width: none !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

def cleanup_chroma_folder(path):
    def _cleanup():
        shutil.rmtree(path, ignore_errors=True)
        print(f"🧹 Deleted Chroma folder: {path}")
    return _cleanup

def streamlit_ui():
    st.title("Quanta: Research-Aware Assistant")

    # -- Initialize session state for file and persistence --
    if "files" not in st.session_state:
        st.session_state["files"] = None

    if "persist_directory" not in st.session_state:
        st.session_state["persist_directory"] = None

    if "collection_name" not in st.session_state:
        st.session_state["collection_name"] = f"collection_{uuid.uuid4().hex[:8]}"

    if "quanta" not in st.session_state:
        st.session_state["quanta"] = None

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # -- Sidebar file upload --
    st.sidebar.header("Upload PDF/DOCX/TXT Files")
    # uploader_key = f"quanta_file_uploader_{st.session_state.get('some_unique_id', 'default')}"

    files = st.sidebar.file_uploader("Upload Documents", type=["pdf", "txt"], accept_multiple_files=True, key="quanta_file_uploader")

    if files:
        if files != st.session_state["files"]:
            st.session_state["files"] = files
            # Generate new persist_dir and collection for new upload
            st.session_state["persist_directory"] = None
            st.session_state["collection_name"] = f"collection_{uuid.uuid4().hex[:8]}"
            st.session_state["quanta"] = None # reset instance
            st.session_state["chat_history"] = [] # reset chat history for new doc
    else:
        st.session_state["files"] = None

    # -- Preprocessing --
    if st.button("Preprocess Document"):
        if st.session_state["files"] is None:
            st.warning("Please upload a document first.")
        else:
            try:
                with st.spinner("Preprocessing documents..."):
                    quanta = Quanta(
                        collection_name=st.session_state["collection_name"]
                    )
                    # Convert UploadedFile objects to a usable list
                    file_list = [f for f in st.session_state["files"] if hasattr(f, "read") and hasattr(f, "name")]

                    # Save uploaded files to temp files before passing to Quanta
                    temp_file_paths = []
                    for f in file_list:
                        suffix = os.path.splitext(f.name)[1]
                        f.seek(0)  # Ensure we read from the start of the file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp.flush()
                            temp_file_paths.append(tmp.name)
                    quanta.preprocess_pipeline(temp_file_paths)
                    st.session_state["quanta"] = quanta

                    # register cleanup to run after Streamlit exits
                    if hasattr(quanta, "persist_directory") and quanta.persist_directory:
                        atexit.register(cleanup_chroma_folder(quanta.persist_directory))

                    st.success("✅ Document processed and indexed!")
            except Exception as e:
                st.error(f"❌ Error during preprocessing: {e}")

    # -- User Query (Question) Input --
    st.subheader("Ask a Question")

    documents_uploaded = st.session_state.get("files") is not None
    documents_processed = st.session_state.get("quanta") is not None
    ready_for_questions = documents_uploaded and documents_processed
    
    user_query = st.text_input("What would you like to ask about the document(s)?", disabled=not ready_for_questions)

    if st.button("Get Answer", disabled=not ready_for_questions):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            quanta = st.session_state.get("quanta")
            if quanta is None:
                st.error("Please preprocess a document first.")
                st.stop()

            try:
                with st.spinner("Thinking..."):
                    answer = quanta.query_pipeline(user_query)

                # Save Q&A to chat history
                st.session_state["chat_history"].append({
                    "question": user_query,
                    "answer": answer
                })

                st.markdown("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")

    # -- Display Chat History --
    if st.session_state["chat_history"]:
        st.markdown("---")
        with st.expander("**Chat History**", expanded=False): # the expanded stores in default state
            for i, entry in enumerate((st.session_state["chat_history"])):
                st.markdown(f"**Q{i+1}:** {entry['question']}")
                st.markdown(f"**A{i+1}:** {entry['answer']}")

if __name__ == "__main__":
    streamlit_ui()