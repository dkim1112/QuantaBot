import streamlit as st
from src.core.quanta import Quanta

def streamlit_ui():
    st.title("Quanta-Bot for Researchers.")

    quanta = Quanta()

    file = st.file_uploader(
        "Upload one or more documents (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )

    query = st.text_input("Enter your query to Quanta!")

    if query:
        st.write("Processing...")
        if file and (
            "summarize" in query.lower()
            or "summary" in query.lower()
            or "summaries" in query.lower()
        ):
            summary = quanta.summary_tool(file)
            st.write("Here is your summary:")
            st.write(summary)
        else:
            retrieved_info = quanta.query_chain(query)
            st.write(retrieved_info)
