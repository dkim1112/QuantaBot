from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
import logging

# Custom loader that uses TextLoader internally and adds robust error handling.
class CustomTextLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        documents = []
        try:
            loader = TextLoader(file_path=self.file_path, autodetect_encoding=True)
            documents = list(loader.lazy_load())
        except Exception as e:
            logging.error(f"Error loading {self.file_path} with TextLoader: {e}")
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents = [Document(page_content=content, metadata={"source": self.file_path})]
            except Exception as fallback_error:
                logging.error(f"Fallback loading failed for {self.file_path}: {fallback_error}")
                raise RuntimeError(f"Could not load {self.file_path}")
        return documents