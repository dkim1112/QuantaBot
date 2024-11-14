from PyPDF2 import PdfReader
from langchain.docstore.document import Document

# Had to create a custom PDF Loader because the PyPDFLoader itself didnt work.
# Compatibility issue with Streamlit.
class CustomPDFLoader:
    def __init__(self, file_obj):
        self.file_obj = file_obj

    def load(self):
        """Load the PDF content from a file-like object."""
        reader = PdfReader(self.file_obj)
        pages = [page.extract_text() for page in reader.pages]
        documents = [
            Document(page_content=page, metadata={"source": self.file_obj.name})
            for page in pages
        ]
        return documents
