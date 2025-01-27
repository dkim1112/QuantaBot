import re
import ast
from langchain.docstore.document import Document

# STREAMLIT Ver.
def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\\s+", " ", text)
    text = re.sub(r"[\\n\\r]+", " ", text)  # Remove line breaks
    text = re.sub(r"[^\\w\\s.,]", "", text)  # Remove special characters
    return text.strip()

def str_to_document(text: str):
    if " metadata=" in text:
        # Normal processing
        page_content_part, metadata_part = text.split(" metadata=", 1)
        page_content = page_content_part.split("page_content", 1)[1].strip("'")
        metadata = ast.literal_eval(metadata_part)
    else:
        # Handle missing metadata gracefully
        print("Warning: Missing metadata. Defaulting to empty metadata.")
        page_content = text.strip()
        metadata = {}

    return Document(page_content=page_content, metadata=metadata)