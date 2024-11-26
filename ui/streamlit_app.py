import streamlit as st
from src.core.quanta import Quanta

def streamlit_ui():
    # Set up the Streamlit app title
    st.title("Quanta-Bot [In-Development]")

    # Initialize Quanta instance
    quanta = Quanta()

    # Initialize messages in session state if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "file" not in st.session_state:
        st.session_state["file"] = None

    # File uploader to allow document uploads
    file = st.file_uploader(
        "Upload one or more documents (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader_unique_key"  # Unique key to prevent duplicate error
    )

    # Save uploaded file(s) to session state
    st.session_state["file"] = file if file else None

    # Display a warning if no file is uploaded
    if not st.session_state["file"]:
        st.warning("Please upload a document to start your chat.")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initialize the chat input
    prompt = None
    if st.session_state["file"]:
        # Enable chat input if a file is uploaded
        prompt = st.chat_input("Ask Quanta something...", key="chat_input_enabled")
    else:
        # Display disabled chat input
        prompt = st.chat_input("Ask Quanta something...", disabled=True, key="chat_input_disabled")

    # Process user input
    if prompt:  # Only proceed if the user enters something
        # **Check file upload status dynamically**
        if not st.session_state["file"]:
            # Warn the user if no file is uploaded at query time
            with st.chat_message("assistant"):
                response = "Please upload a document before asking a question."
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            return  # Stop further processing

        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the query with Quanta and get the response
        if "summarize" in prompt.lower():
            # Use Quanta to summarize the file if the query includes "summarize"
            response = quanta.summary_tool(st.session_state["file"])
        else:
            # Use Quanta’s query chain for other queries
            response = quanta.query_chain(st.session_state["file"], prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant message to session state
        st.session_state.messages.append({"role": "assistant", "content": response})


streamlit_ui()
