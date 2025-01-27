import streamlit as st
from src.core.quanta import Quanta

def streamlit_ui():
    # Set up the Streamlit app title
    st.title("Quanta-Bot [In-Development]")

    # Initialize Quanta instance
    quanta = Quanta()

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "file" not in st.session_state:
        st.session_state["file"] = None
    if "preprocessed" not in st.session_state:
        st.session_state["preprocessed"] = False

    # File uploader section
    file = st.file_uploader(
        "Upload one or more documents (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader_unique_key"
    )

    # Save uploaded file(s) to session state
    if file != st.session_state["file"]:
        st.session_state["file"] = file
        st.session_state["preprocessed"] = False  # Reset preprocessed state when new files are uploaded

    # Preprocessing section
    if st.session_state["file"]:
        if not st.session_state["preprocessed"]:
            if st.button("Preprocess Documents"):
                try:
                    with st.spinner("Preprocessing documents..."):
                        quanta.preprocess_pipeline(st.session_state["file"])
                    st.session_state["preprocessed"] = True
                    st.success("Documents preprocessed successfully!")
                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")
        else:
            st.success("Documents are preprocessed and ready for querying!")
    else:
        st.warning("Please upload a document to start your chat.")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input section
    if st.session_state["file"] and st.session_state["preprocessed"]:
        prompt = st.chat_input("Ask Quanta something...", key="chat_input_enabled")
    else:
        prompt = st.chat_input(
            "Please upload and preprocess documents first...", 
            disabled=True, 
            key="chat_input_disabled"
        )

    # Process user input
    if prompt:
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            # Process the query
            with st.spinner("Processing your query..."):
                if "summarize" in prompt.lower():
                    response = quanta.summary_tool(st.session_state["file"])
                else:
                    response = quanta.query_pipeline(prompt)

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)

            # Add assistant message to session state
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            with st.chat_message("assistant"):
                error_message = f"Error processing query: {str(e)}"
                st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

streamlit_ui()