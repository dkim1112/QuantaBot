from PyPDF2 import PdfReader
from langchain.docstore.document import Document
import logging

# Custom loader that uses PdfLoader internally and adds robust error handling.
class CustomPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            reader = PdfReader(self.file_path)
            pages = [page.extract_text() for page in reader.pages]
            documents = [
                Document(page_content=page, metadata={"source": self.file_path})
                for page in pages
            ]
            return documents
        except Exception as e:
            logging.error(f"Error loading PDF file {self.file_path}: {e}")
            raise RuntimeError(f"Failed to load PDF file {self.file_path}: {e}")

 