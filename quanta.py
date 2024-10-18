import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from io import BytesIO  # Needed to handle file-like objects
from langchain.prompts import PromptTemplate
from pathlib import Path
from PyPDF2 import PdfReader  # Using PyPDF2 for PDF reading
import openai


class CustomPDFLoader:
    def __init__(self, file_obj):
        self.file_obj = file_obj

    def load(self):
        """Load the PDF content from a file-like object."""
        reader = PdfReader(self.file_obj)
        pages = [page.extract_text() for page in reader.pages]
        documents = [{"page_content": page} for page in pages]
        return documents


class RAGAgent:
    def __init__(self, openai_api_key):
        """Initialize the RAGAgent with the retriever and language model."""
        # Set OpenAI API key using OpenAI's Python client
        openai.api_key = openai_api_key

        # Initialize OpenAI language model without passing openai_api_key directly
        self.llm = OpenAI(temperature=0)
        self.document_store = self.get_doc_store()
        self.retriever = self.document_store.as_retriever()

    def get_doc_store(self):
        """Initialize ChromaDB as the document store for retrieval."""
        return Chroma(
            persist_directory="chroma_db", embedding_function=OpenAIEmbeddings()
        )

    def write_documents(self, file):
        """Load and write the documents into the ChromaDB document store."""
        if file.name.endswith(".docx"):
            loader = TextLoader(file.name)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file.name)
        elif file.name.endswith(".pdf"):
            loader = CustomPDFLoader(file)

        documents = loader.load()
        self.document_store.add_documents(documents)

    def chunk_documents(self, file):
        """Chunk documents for further processing (summarization)."""
        if file.name.endswith(".docx"):
            loader = TextLoader(file.name)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file.name)
        elif file.name.endswith(".pdf"):
            loader = CustomPDFLoader(file)

        documents = loader.load()

        def split_into_chunks(text, chunk_size=3000):
            words = text.split()
            return [
                " ".join(words[i : i + chunk_size])
                for i in range(0, len(words), chunk_size)
            ]

        chunked_docs = []
        for doc in documents:
            content = doc.get("content") or doc.get("page_content")
            if content:
                chunked_docs.extend(split_into_chunks(content))
        return chunked_docs

    def query_chain(self, query):
        """Run a query using the LangChain-based QA chain with ChromaDB."""
        qa_chain = RetrievalQA(llm=self.llm, retriever=self.retriever)
        result = qa_chain.run(query)
        return result

    def query_router(self, query):
        """Decide whether the query needs summarization, retrieval, or a simple response."""
        system_prompt = """You are a professional decision-making query router bot for a chatbot system.
        You decide whether a user’s query requires:
        1. A summary (reply with "1")
        2. Retrieval of extra information from a vector database (reply with "2")
        3. A simple greeting/gratitude/salutation response (reply with "3")."""

        instruction = f"{system_prompt}\nUser query: {query}"
        response = self.llm(instruction)
        return response.strip()

    def map_summarizer(self, query, chunk):
        """Summarize a chunk of text based on a user’s query."""
        prompt_template = PromptTemplate(
            input_variables=["query", "chunk"],
            template="""
            You are a professional summarizer for a chatbot system.
            You are responsible for summarizing a chunk of text according to a user’s query.
            User query: {query}
            Text chunk to summarize: {chunk}
            Provide a concise and accurate summary based on the user’s query.
            """,
        )
        prompt = prompt_template.format(query=query, chunk=chunk)
        summary = self.llm(prompt)
        return summary

    def simple_responder(self, query):
        """Generate simple responses such as greetings or follow-ups."""
        simple_prompt = (
            f"Please generate a simple response to the following query: '{query}'"
        )
        response = self.llm(simple_prompt)
        return response

    def handle_intent(self, intent, query, chunk=None):
        """Handle different intents like summarization or retrieval."""
        if intent == 1:  # Summarization intent
            return self.map_summarizer(query, chunk)
        elif intent == 2:  # Retrieval intent
            return self.query_chain(query)
        else:
            return self.simple_responder(query)

    def reduce_summarizer(self, summaries):
        """Reduce multiple summaries into a final, concise summary."""
        combined_summaries = " ".join(summaries)
        prompt_template = PromptTemplate(
            input_variables=["summaries"],
            template="""
            You are a professional summarizer for a chatbot system.
            You are responsible for reducing multiple summaries into a single concise one.
            Here are the summaries: {summaries}
            Provide a concise, accurate summary.
            """,
        )
        prompt = prompt_template.format(summaries=combined_summaries)
        reduced_summary = self.llm(prompt)
        return reduced_summary

    def summary_tool(self, file):
        """Summarize an entire document by splitting it into chunks, summarizing, and reducing."""
        # Step 1: Chunk the document
        chunked_docs = self.chunk_documents(file)

        # Step 2: Summarize each chunk
        summaries = [
            self.map_summarizer("Summarize this document", chunk)
            for chunk in chunked_docs
        ]

        # Step 3: Reduce multiple summaries into a final summary
        final_summary = self.reduce_summarizer(summaries)

        return final_summary

    def context_tool(self, query):
        """Retrieve relevant context for a query from the document store."""
        retriever = self.document_store.as_retriever()
        context_docs = retriever.get_relevant_documents(query)

        # Format the documents into a concise context
        prompt_template = PromptTemplate(
            input_variables=["docs"],
            template="""
            You are a professional assistant for a chatbot system.
            Based on the following documents, provide a relevant context for the user's query.
            Documents: {docs}
            """,
        )
        docs_content = " ".join([doc.page_content for doc in context_docs])
        prompt = prompt_template.format(docs=docs_content)

        context_response = self.llm(prompt)
        return context_response


# Streamlit Interface to interact with the RAGAgent
def streamlit_ui():
    st.title("RAG-based Summarization and Retrieval System")

    # Ask the user for their OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    if openai_api_key:
        # Initialize the agent with the provided OpenAI API key
        rag_agent = RAGAgent(openai_api_key=openai_api_key)

        # File uploader
        file = st.file_uploader(
            "Upload your document (txt, pdf, docx)", type=["txt", "pdf", "docx"]
        )

        # Query input
        query = st.text_input("Enter your query for summarization or retrieval:")

        # Buttons for summarization and retrieval
        if st.button("Summarize Document") and file:
            st.write("Summarizing...")
            summary = rag_agent.summary_tool(file)
            st.write("Summary of the document:")
            st.write(summary)

        if st.button("Retrieve Information") and query:
            st.write("Retrieving information...")
            retrieved_info = rag_agent.query_chain(query)
            st.write("Retrieved Information:")
            st.write(retrieved_info)


if __name__ == "__main__":
    streamlit_ui()


# TODO : Add more error handling and logging
# TODO : Add more comments and docstrings
# TODO : Add streaming responses
# TODO : Add self rag loop through invoking agent until answer is found
# TODO : Add URL abilities
