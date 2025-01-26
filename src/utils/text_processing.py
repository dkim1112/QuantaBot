import re
import ast
from langchain.docstore.document import Document

def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    return text

# STREAMLIT Ver.
def str_to_document(text: str):
    page_content_part, metadata_part = text.split(" metadata=")
    page_content = page_content_part.split("page_content", 1)[1].strip("'")
    metadata = ast.literal_eval(metadata_part)
    return Document(page_content=page_content, metadata=metadata)

# TERMINAL Ver.
# def str_to_document(text: str):

#     # Check if metadata is present
#     if " metadata=" in text:
#         # Normal processing
#         page_content_part, metadata_part = text.split(" metadata=", 1)
#         page_content = page_content_part.split("page_content", 1)[1].strip("'")
#         try:
#             metadata = ast.literal_eval(metadata_part)
#         except Exception as e:
#             raise ValueError(f"Failed to parse metadata: {e}")
#     else:
#         # Handle missing metadata
#         print("Warning: Missing metadata.")
#         page_content = text.strip()
#         metadata = {}

#     return Document(page_content=page_content, metadata=metadata)
