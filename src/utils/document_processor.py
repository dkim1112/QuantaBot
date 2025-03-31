from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from ..loaders.pdf_loader import CustomPDFLoader
from ..loaders.txt_loader import CustomTextLoader
from ..loaders.docs_loader import CustomDocsLoader
from .text_processing import preprocess_text
import os
    
class DocumentProcessor:
    @staticmethod
    def chunk_documents(files, document_store, chunk_size=500, chunk_overlap=100):

        def _flatten(lst):
            for item in lst:
                if isinstance(item, list):
                    yield from _flatten(item)
                else:
                    yield item

        files = list(_flatten(files))

        all_docs = []
        for file in files:
            # Check if file is a string path or a file object
            if isinstance(file, str):
                file_path = file
                file_name = os.path.basename(file_path)
            else:
                raise ValueError("Expected file path as string, got something else.")

            # Load and write the documents into the ChromaDB document store.
            if file_name.endswith(".docx"):
                loader = CustomDocsLoader(file_path)
                documents = loader.load()
            elif file_name.endswith(".txt"):
                loader = CustomTextLoader(file_path)
                documents = loader.load()
            elif file_name.endswith(".pdf"):
                loader = CustomPDFLoader(file_path)
                documents = loader.load()
            # Defensive programming
            else:
                raise ValueError(f"Unsupported file type: {file_name}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " "], # Prioritize meaningful splits
            )
            raw_chunks = text_splitter.split_documents(documents)
            processed_chunks = DocumentProcessor.adjust_chunk_boundaries(raw_chunks)

            # Preprocess each chunk
            for chunk in processed_chunks:
                processed_text = preprocess_text(chunk.page_content) # Preprocess chunk text
                metadata = chunk.metadata # Retain metadata for context
                document = Document(page_content=processed_text, metadata=metadata)
                all_docs.append(document)

            # Add documents to the document store
            document_store.add_documents(all_docs)
            
        # We are returning a full set of chunks that combined all the files uploaded into one.
        return all_docs
    
    @staticmethod
    def adjust_chunk_boundaries(chunks):
        """Ensures chunks end at full sentences to avoid cut-off issues."""
        adjusted_chunks = []
        for chunk in chunks:
            text = chunk.page_content.strip()
            last_period = max(text.rfind("."), text.rfind("?"), text.rfind("!")) # Find last full stop
            if last_period != -1 and last_period < len(text) - 5: # Ensure the period is meaningful
                text = text[:last_period + 1] # Trim at the last full stop
            
            # Recreate the Document instance to preserve metadata
            adjusted_chunk = Document(page_content=text, metadata=chunk.metadata)
            adjusted_chunks.append(adjusted_chunk)
        return adjusted_chunks
