from langchain.docstore.document import Document
from docx import Document as DocxDocument
import logging

# Custom loader that uses DocsLoader internally and adds robust error handling.
class CustomDocsLoader:
    
    # STREAMLIT Ver.
    def __init__(self, file_obj):
        self.file_obj = file_obj

    def load(self):
        try:
            # Open the .docx file using python-docx
            doc = DocxDocument(self.file_obj)
            # Extract text from paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            content = "\n".join(paragraphs)
            # Create LangChain Document object
            return [Document(page_content=content, metadata={"source": self.file_obj.name})]
        except Exception as e:
            logging.error(f"Error loading .docx file {self.file_obj.name}: {e}")
            raise RuntimeError(f"Could not load .docx file {self.file_obj.name}")

    # TERMINAL Ver.
    # def __init__(self, file_path):
    #     self.file_path = file_path

    # def load(self):
    #     try:
    #         # Open the .docx file using python-docx
    #         doc = DocxDocument(self.file_path)
    #         # Extract text from paragraphs
    #         paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    #         content = "\n".join(paragraphs)
    #         # Create LangChain Document object
    #         return [Document(page_content=content, metadata={"source": self.file_path})]
    #     except Exception as e:
    #         logging.error(f"Error loading .docx file {self.file_path}: {e}")
    #         raise RuntimeError(f"Could not load .docx file {self.file_path}")