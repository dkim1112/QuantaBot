from langchain.text_splitter import NLTKTextSplitter
from ..loaders.pdf_loader import CustomPDFLoader
from ..loaders.txt_loader import CustomTextLoader
from ..loaders.docs_loader import CustomDocsLoader
from .text_processing import preprocess_text, str_to_document

class DocumentProcessor:
    @staticmethod
    def chunk_documents(files, document_store):
        all_docs = []
        for file in files:
            # Load and write the documents into the ChromaDB document store.
            if file.name.endswith(".docx"):
                loader = CustomDocsLoader(file)
                documents = loader.load()
            elif file.name.endswith(".txt"):
                loader = CustomTextLoader(file)
                documents = loader.load()
            elif file.name.endswith(".pdf"):
                loader = CustomPDFLoader(file)
                documents = loader.load()
            # Defensive programming
            else:
                raise ValueError("Unsupported file type")

            # Instead of splitting recursively by character
            # Decided to split by sentence (so that no words get cut off in summary).
            text_splitter = NLTKTextSplitter()
            raw_chunks = text_splitter.split_documents(documents)
            
            chunks = [str(documents) for documents in raw_chunks]
            chunks = [preprocess_text(chunk) for chunk in chunks]
            docs = [str_to_document(chunk) for chunk in chunks]

            # docs is a list of documents. We extend (append) that to a list of all_docs, which has all of it.
            all_docs.extend(docs)

        # Add documents to the document store
        document_store.add_documents(all_docs)
        
        # We are returning a full set of chunks that combined all the files uploaded into one.
        return all_docs