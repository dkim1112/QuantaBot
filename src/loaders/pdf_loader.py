from PyPDF2 import PdfReader
from langchain.docstore.document import Document
import logging

# Custom loader that uses PdfLoader internally and adds robust error handling.
class CustomPDFLoader:
    # STREAMLIT Ver.
    def __init__(self, file_obj):
        self.file_obj = file_obj

    def load(self):
        """Load the PDF content from a file-like object."""
        try:
            reader = PdfReader(self.file_obj)
            pages = [page.extract_text() for page in reader.pages]
            documents = [
                Document(page_content=page, metadata={"source": self.file_obj.name})
                for page in pages
            ]
            return documents
        except Exception as e:
            logging.error(f"Error loading PDF file {self.file_obj.name}: {e}")
            raise RuntimeError(f"Failed to load PDF file {self.file_obj.name}: {e}")
 