import uuid
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

from ..utils.embedding_wrapper import HuggingFaceEmbeddings
from ..utils.document_processor import DocumentProcessor
from src.core.llm import MyOpenAI

# Main component of the Chatbot Quanta.
class Quanta:
    def __init__(self, llm=None, collection_name=None):
        self.llm = llm or MyOpenAI() # Real OpenAI call or own model wrapper
        self.embedding_function = HuggingFaceEmbeddings() # Sentence-BERT wrapper
        self.collection_name = collection_name or f"collection_{uuid.uuid4().hex[:8]}"
        self.document_store = None
        self.persist_directory = None


    def get_doc_store(self):
        self.persist_directory = f"chroma_{self.collection_name}"

        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )
    

    def embed(self, text):
        return self.embedding_function.embed_query(text)
    
    
    def reset_document_store(self):
        # Forget previous connection and delete directory
        self.collection_name = f"collection_{uuid.uuid4().hex[:8]}"
        self.document_store = self.get_doc_store()


    def preprocess_pipeline(self, file_paths, batch_size=10):
        file_paths = [item for sublist in file_paths for item in (sublist if isinstance(sublist, list) else [sublist])]
        self.reset_document_store()
        if self.document_store is None:
            self.document_store = self.get_doc_store()

        all_documents = []
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            documents = DocumentProcessor.chunk_documents(batch_paths, self.document_store)
            embeddings = [self.embed(doc.page_content) for doc in documents]
            for idx, doc in enumerate(documents):
                doc.metadata["embedding"] = str(embeddings[idx])
            self.document_store.add_documents(documents, embeddings=embeddings)
            all_documents.extend(documents)

        return all_documents


    def query_pipeline(self, query, top_n_chunks=5):
        if self.document_store is None:
            self.document_store = self.get_doc_store()
            
        query_embedding = self.embed(query)

        # Depends on which vector DB used, but ChromaDB is cosine similarity by default.
        results = self.document_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=top_n_chunks,
        )

        chunks = [doc.page_content for doc in results]

        # Used for DB setup debugging.
        # print(f"Querying ChromaDB at: {self.persist_dir}")
        # print(f"Query: {query}")
        # print(f"Retrieved {len(results)} results.")

        # for i, doc in enumerate(results[:3]):
        #     print(f"Chunk {i} Preview: {doc.page_content[:200]}")

        return self.generate_response(query, chunks)


    def generate_response(self, query, chunks):
        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Keep in mind that this is most likely a research paper.
            You are a highly knowledgeable assistant. Answer the query below based on the provided context, but you may also use your own knowledge if necessary.
            User query: {query}
            Context: {context}
            Response:
            """,
        )
        context = "\n".join(chunks)
        prompt = prompt_template.format(query=query, context=context)

        return self.llm.invoke(prompt)