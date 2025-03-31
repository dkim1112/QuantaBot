from ui.streamlit_app import streamlit_ui
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    streamlit_ui()

# DONE : Cut down on the time it takes for response
# DONE : Document Summarization: It can summarize documents to provide concise answers or overviews.
# DONE : Follow-up Answers: The quanta can answer follow-up questions based on previous interactions and the current conversation context.
# DONE : Continuous talking (back-and-forth)
# DONE : Allow free talking & asking general questions about the document too.
# DONE : UI Streamlit Revision
# DONE : Implement cosine similarity logic, reduction techniques, and more.
# AND MUCH MORE...