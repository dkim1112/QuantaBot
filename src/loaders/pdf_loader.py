from PyPDF2 import PdfReader
from langchain.docstore.document import Document
import logging

# Custom loader that uses PdfLoader internally and adds robust error handling.
class CustomPDFLoader:
    def __init__(self, file_path, original_filename=None):
        self.file_path = file_path
        self.original_filename = original_filename or file_path.split("/")[-1]

    def load(self):
        try:
            reader = PdfReader(self.file_path)
            documents = []

            for page_num, page in enumerate(reader.pages, start=1):
                page_content = page.extract_text()
                if page_content.strip():  # Only add pages with content
                    document = Document(
                        page_content=page_content,
                        metadata={
                            "source": self.file_path,
                            "page": page_num,
                            "filename": self.original_filename  # Use original filename
                        }
                    )
                    documents.append(document)

            return documents
        except Exception as e:
            logging.error(f"Error loading PDF file {self.file_path}: {e}")
            raise RuntimeError(f"Failed to load PDF file {self.file_path}: {e}")

 