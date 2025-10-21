import uuid
import os
import numpy as np
from typing import List, Optional, Dict, Any

# LangChain imports
from langchain_chroma import Chroma
# Import retrievers from the correct modules in LangChain v0.3+
try:
    from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever, ParentDocumentRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
except ImportError:
    # Fallback for newer versions
    try:
        from langchain_community.retrievers import MultiQueryRetriever, EnsembleRetriever, ParentDocumentRetriever
        from langchain_community.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.retrievers.contextual_compression import ContextualCompressionRetriever
    except ImportError:
        # Manual imports as last resort
        print("Warning: Some retrievers may not be available. Please check your LangChain installation.")
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.retrievers import BM25Retriever

# Custom imports
from ..utils.embedding_wrapper import HuggingFaceEmbeddings
from ..utils.document_processor import DocumentProcessor
from .llm import MyOpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QuantaCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for monitoring and debugging QuantaBot operations."""

    def __init__(self):
        self.retrieval_stats = []
        self.query_count = 0

    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs):
        self.query_count += 1

    def on_retriever_end(self, documents: List[Document], **kwargs):
        doc_count = len(documents)
        avg_length = np.mean([len(doc.page_content) for doc in documents]) if documents else 0
        self.retrieval_stats.append({"doc_count": doc_count, "avg_length": avg_length})

    def get_stats(self):
        return {
            "total_queries": self.query_count,
            "retrieval_stats": self.retrieval_stats
        }


class LangChainQuantaBot:
    def __init__(self, llm=None, collection_name=None):
        self.llm = llm or MyOpenAI()
        self.embedding_function = HuggingFaceEmbeddings()
        self.collection_name = collection_name or f"collection_{uuid.uuid4().hex[:8]}"
        self.persist_directory = f"chroma_{self.collection_name}"

        # LangChain components
        self.vectorstore = None
        self.retriever = None
        self.retrieval_chain = None
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True
        )

        # Monitoring
        self.callback_handler = QuantaCallbackHandler()

        # Storage for parent-document retrieval
        self.docstore = InMemoryStore()

        # Text splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Large chunks for parent documents
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
            keep_separator=True,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,   # Smaller chunks for retrieval
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "],
            keep_separator=True,
        )

    def setup_vectorstore(self):
        """Initialize the vector store. Utilizes ChromaDB"""
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )
        return self.vectorstore

    def create_advanced_retriever(self) -> ContextualCompressionRetriever:
        if self.vectorstore is None:
            self.setup_vectorstore()

        # 1. Create parent-document retriever for better context
        parent_retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

        # 2. Create multi-query retriever for query expansion
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=parent_retriever,
            llm=self.llm,
            include_original=True
        )

        # 3. Add cross-encoder reranking
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(
            model=cross_encoder_model,
            top_n=12
        )

        # 4. Create contextual compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=multi_query_retriever,
            callbacks=[self.callback_handler]
        )

        return compression_retriever

    def setup_ensemble_retriever(self, documents: List[Document]) -> EnsembleRetriever:
        """Set up ensemble retriever combining semantic and keyword search."""

        # Create BM25 retriever from documents
        texts = [doc.page_content for doc in documents]
        bm25_retriever = BM25Retriever.from_texts(texts)
        bm25_retriever.k = 6  # Number of documents to retrieve

        # Create semantic retriever (MMR (includes cosine sim.))
        semantic_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )

        # Combine retrievers with weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.7, 0.3], # Favor semantic search slightly
            callbacks=[self.callback_handler]
        )

        return ensemble_retriever

    def format_documents(self, documents):
        """Custom document formatter that includes filename and page info for proper citations."""
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            filename = doc.metadata.get("filename", "Unknown Document")
            page = doc.metadata.get("page", "N/A")

            # Skip documents with empty metadata or unknown filenames
            if not doc.metadata or filename == "Unknown Document":
                # Still include the content but without source info for the LLM
                formatted_doc = f"{doc.page_content}\n"
            else:
                if page != "N/A":
                    source_info = f"[Source: {filename}, page {page}]"
                else:
                    source_info = f"[Source: {filename}]"
                formatted_doc = f"{source_info}\n{doc.page_content}\n"

            formatted_docs.append(formatted_doc)

        return "\n---\n".join(formatted_docs)

    def create_rag_chain(self):
        """Create a proper RAG chain using LangChain's create_retrieval_chain."""

        system_prompt = """
        You are a highly capable AI research assistant specializing in academic analysis.
        You have access to carefully selected and reranked document chunks that are highly relevant to the user's query.

        Your task is to provide comprehensive, accurate responses by synthesizing information from ALL provided context.

        CRITICAL INSTRUCTIONS:
        - The context has been processed through advanced retrieval including query expansion, semantic search, keyword search, and relevance reranking
        - Each piece of context is valuable - use information from ALL context pieces when relevant
        - Provide detailed, specific answers with examples, numbers, and quotes when available
        - Structure your response clearly with headings or bullet points when appropriate
        - If information is insufficient, clearly state what's missing rather than guessing
        - ALWAYS include citations for your sources using the exact source information provided in the context

        YOUR RESPONSE SHOULD FOLLOW THESE PROTOCOLS:
        1. Context Analysis:
        - Carefully analyze the provided context to understand its relevance to the query.
        - Identify key themes, arguments, or data points that directly relate to the query.

        2. Query Understanding:
        - Ensure that the query is fully understood. If the query is ambiguous, identify potential interpretations and choose the most logical one based on the context.
        - Address the query based on its scope – whether it requires a direct answer, an in-depth analysis, or a synthesis of the information.

        3. Answer Construction:
        - Formulate a clear, logically structured response using precise and formal academic language.
        - The answer should read as a naturally articulated, stand-alone academic explanation - DO NOT INCLUDE ANY META-COMMENTARY (e.g., "the context shows," "the query asks," "based on the provided information," etc.).
        - When sufficient information is available, present a detailed and well-reasoned explanation, incorporating specific insights, data points, and arguments where relevant.
        - Be SPECIFIC with the answers, referring to even the smallest details in the context if relevant.
        - In cases where critical information is missing, enhance the answer using relevant, domain-specific knowledge.
        - Distinguish clearly when supplemental information originates from general scholarly understanding rather than the provided materials.
        - Use definitions, classifications, or examples to clarify technical terms or complex concepts when appropriate.
        - Use appropriate styling such as bullet points or numbered lists for clarity, especially when presenting multiple points or steps in an argument.

        4. Citation Requirements:
        - Each context section begins with source information in the format [Source: filename, page X] or [Source: filename]
        - Use EXACTLY this source information when citing - do not create your own citation format
        - For EVERY piece of information you use from the provided context, include the exact source citation
        - When you quote directly, use quotation marks and include the citation immediately after: "quoted text" [Source: filename, page X]
        - At the end of your response, include a "References" section listing all sources cited

        5. Exceptions and Clarifications:
        - If the query or context contains sensitive or ethical considerations, handle these with appropriate sensitivity and discretion.
        - In the case of highly technical or domain-specific subjects, clarify complex terms and concepts to ensure understanding.

        Context: {context}

        Question: {input}

        Detailed Response (with citations):
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt)
        ])

        # Create custom document combination chain with our formatter
        from langchain_core.runnables import RunnableLambda

        def format_docs_chain(inputs):
            """Custom chain that formats documents with source info before LLM processing."""
            documents = inputs["context"]
            formatted_context = self.format_documents(documents)

            # Create the final prompt input
            return {
                "context": formatted_context,
                "input": inputs["input"]
            }

        # Create the combine documents chain with custom formatting
        combine_docs_chain = (
            RunnableLambda(format_docs_chain)
            | prompt
            | self.llm
        )

        # Create retrieval chain
        self.retrieval_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=combine_docs_chain
        )

        return self.retrieval_chain

    def preprocess_documents(self, file_paths: List[str], batch_size: int = 10, file_mapping: dict = None):
        """Process documents using LangChain's parent-document pattern."""

        file_paths = [item for sublist in file_paths for item in (sublist if isinstance(sublist, list) else [sublist])]

        # Reset document store and vector store
        self.docstore = InMemoryStore()
        self.setup_vectorstore()

        # Process documents in batches
        all_documents = []

        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i:i + batch_size]
            documents = DocumentProcessor.chunk_documents(
                batch_paths,
                self.vectorstore,
                chunk_size=1200,  # Use our optimized chunk size
                chunk_overlap=200,
                file_mapping=file_mapping
            )
            all_documents.extend(documents)

        # Create advanced retriever
        if len(all_documents) > 20:  # Use ensemble for larger document sets
            # Set up parent-document retriever first
            parent_retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
            )

            # Add documents to parent-document retriever
            parent_retriever.add_documents(all_documents)

            # Create ensemble with parent retriever
            self.retriever = self.setup_ensemble_retriever(all_documents)
        else:
            # Use compression retriever for smaller sets
            self.retriever = self.create_advanced_retriever()

            # Add documents directly
            parent_retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
            )
            parent_retriever.add_documents(all_documents)

        # Create the RAG chain
        self.create_rag_chain()
        return all_documents

    def query(self, question: str) -> dict:
        """Query the LangChain RAG system with conversation memory."""

        if self.retrieval_chain is None:
            raise ValueError("Please preprocess documents first using preprocess_documents()")

        # Get conversation history for context (last 10 messages)
        chat_history = self.memory.chat_memory.messages[-10:]

        # Create enhanced input with conversation context
        enhanced_input = {
            "input": question,
            "chat_history": chat_history
        }

        # Run the retrieval chain
        response = self.retrieval_chain.invoke(enhanced_input)

        # Save to memory
        self.memory.save_context(
            {"input": question},
            {"output": response["answer"]}
        )

        # Return both answer and source documents for citations
        return {
            "answer": response["answer"],
            "source_documents": response.get("context", [])
        }

    def get_retrieval_stats(self):
        """Get retrieval statistics for debugging."""
        return self.callback_handler.get_stats()

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()

    def reset_document_store(self):
        """Reset the document store and create new collection."""
        self.collection_name = f"collection_{uuid.uuid4().hex[:8]}"
        self.persist_directory = f"chroma_{self.collection_name}"
        self.docstore = InMemoryStore()
        self.vectorstore = None
        self.retriever = None
        self.retrieval_chain = None
        self.memory.clear()