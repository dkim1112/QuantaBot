import streamlit as st
from src.core.quanta import Quanta  # Make sure to import your Quanta class

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
    if file:
        st.session_state["file"] = file

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User prompt input through a chat input widget
    if prompt := st.chat_input("Ask Quanta something..."):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process the query with Quanta and get the response
        if st.session_state["file"] and ("summarize" in prompt.lower()):
            # Use Quanta to summarize the file if it’s uploaded and query includes "summarize"
            response = quanta.summary_tool(st.session_state["file"])
        else:
            # Use Quanta’s query chain for other queries
            response = quanta.query_chain(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant message to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

streamlit_ui()