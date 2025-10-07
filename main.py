from ui.langchain_streamlit_app import streamlit_ui
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    streamlit_ui()