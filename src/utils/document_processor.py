from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ..loaders.pdf_loader import CustomPDFLoader
from ..loaders.txt_loader import CustomTextLoader
from ..loaders.docs_loader import CustomDocsLoader
from .text_processing import preprocess_text
import os
    
class DocumentProcessor:
    @staticmethod
    def chunk_documents(files, document_store, chunk_size=1200, chunk_overlap=200, file_mapping=None):
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

                # Get original filename from mapping if available
                original_filename = None
                if file_mapping and file_path in file_mapping:
                    original_filename = file_mapping[file_path]
            else:
                raise ValueError("Expected file path as string, got something else.")

            # Load and write the documents into the ChromaDB document store.
            if file_name.endswith(".docx"):
                loader = CustomDocsLoader(file_path, original_filename)
                documents = loader.load()
            elif file_name.endswith(".txt"):
                loader = CustomTextLoader(file_path, original_filename)
                documents = loader.load()
            elif file_name.endswith(".pdf"):
                loader = CustomPDFLoader(file_path, original_filename)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {file_name}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "], # Prioritize meaningful splits
                keep_separator=True, # Preserve separators for context
            )
            raw_chunks = text_splitter.split_documents(documents)
            processed_chunks = DocumentProcessor.adjust_chunk_boundaries(raw_chunks)

            # Preprocess each chunk
            for chunk in processed_chunks:
                processed_text = preprocess_text(chunk.page_content) # Preprocess chunk text
                metadata = chunk.metadata # Retain metadata for context
                document = Document(page_content=processed_text, metadata=metadata)
                all_docs.append(document)

        # Add all documents to the document store
        document_store.add_documents(all_docs)

        # We are returning a full set of chunks that combined all the files uploaded into one.
        return all_docs
    
    @staticmethod
    def adjust_chunk_boundaries(chunks):
        """Ensures chunks end at full sentences and maintains semantic coherence."""
        adjusted_chunks = []
        for chunk in chunks:
            text = chunk.page_content.strip()

            # Find the best sentence ending position
            sentence_endings = []
            for pattern in [". ", "! ", "? "]:
                pos = text.rfind(pattern)
                if pos != -1 and pos < len(text) - 2:  # Not at the very end
                    sentence_endings.append(pos + 1)  # Include the punctuation

            if sentence_endings:
                # Choose the latest sentence ending that's not too close to the end
                best_ending = max(sentence_endings)
                if best_ending > len(text) * 0.8:  # If it's more than 80% through the text
                    text = text[:best_ending + 1]

            # Only keep chunks that have meaningful content
            if len(text.strip()) > 50:  # Minimum chunk size
                adjusted_chunk = Document(page_content=text, metadata=chunk.metadata)
                adjusted_chunks.append(adjusted_chunk)
        return adjusted_chunks
