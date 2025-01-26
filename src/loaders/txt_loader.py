from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
import logging

# Custom loader that uses TextLoader internally and adds robust error handling.
class CustomTextLoader:
    # STREAMLIT Ver.
    def __init__(self, file_obj):
        self.file_obj = file_obj

    def load(self):
        documents = []
        try:
            # Create a temporary TextLoader with autodetect encoding enabled
            loader = TextLoader(file_path=self.file_obj.name, autodetect_encoding=True)
            documents = list(loader.lazy_load())
        except Exception as e:
            logging.error(f"Error loading {self.file_obj.name} with TextLoader: {e}")
            # Fallback: Read the file manually and wrap it as a Document
            try:
                content = self.file_obj.read().decode("utf-8")
                documents = [Document(page_content=content, metadata={"source": self.file_obj.name})]
            except Exception as fallback_error:
                logging.error(f"Fallback loading failed for {self.file_obj.name}: {fallback_error}")
                raise RuntimeError(f"Could not load {self.file_obj.name}")
        return documents

    # TERMINAL Ver.
    # def __init__(self, file_path):
    #     self.file_path = file_path

    # def load(self):
    #     documents = []
    #     try:
    #         # Create a temporary TextLoader with autodetect encoding enabled
    #         loader = TextLoader(file_path=self.file_path, autodetect_encoding=True)
    #         documents = list(loader.lazy_load())
    #     except Exception as e:
    #         logging.error(f"Error loading {self.file_path} with TextLoader: {e}")
    #         # Fallback: Read the file manually and wrap it as a Document
    #         try:
    #             content = self.file_obj.read().decode("utf-8")
    #             documents = [Document(page_content=content, metadata={"source": self.file_path})]
    #         except Exception as fallback_error:
    #             logging.error(f"Fallback loading failed for {self.file_path}: {fallback_error}")
    #             raise RuntimeError(f"Could not load {self.file_path}")
    #     return documents

