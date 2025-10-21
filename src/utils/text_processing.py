import re
import ast
from langchain_core.documents import Document

def preprocess_text(text):
    # Preserve important academic formatting patterns
    text = re.sub(r"\n{3,}", "\n\n", text)  # Keep paragraph breaks but limit to double
    text = re.sub(r"[ \t]+", " ", text)  # Normalize spaces and tabs

    # Preserve important punctuation and symbols while removing excessive noise
    text = re.sub(r"[^\w\s.,!?()\"';:/@#$%&*+\-=\[\]{}|\\<>~`]", " ", text)
    # Clean up excessive whitespace but preserve structure
    text = re.sub(r"[ ]{3,}", "  ", text)  # Limit excessive spaces to double
    text = re.sub(r"\n[ ]+", "\n", text)  # Remove spaces after newlines
    text = re.sub(r"[ ]+\n", "\n", text)  # Remove spaces before newlines

    # Preserve some formatting for better context
    text = re.sub(r"(\d+\.)\s*", r"\1 ", text)  # Normalize numbered lists
    text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)  # Ensure space after sentences

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