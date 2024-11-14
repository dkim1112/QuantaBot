import re
import ast
from langchain.docstore.document import Document

def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    return text

def str_to_document(text: str):
    page_content_part, metadata_part = text.split(" metadata=")
    page_content = page_content_part.split("page_content", 1)[1].strip("'")
    metadata = ast.literal_eval(metadata_part)
    return Document(page_content=page_content, metadata=metadata)
