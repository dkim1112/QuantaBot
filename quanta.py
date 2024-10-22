import streamlit as st
import re
import ast

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import retrieval_qa
from langchain_openai import OpenAI
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

from io import BytesIO  # Needed to handle file-like objects
from langchain.prompts import PromptTemplate
from pathlib import Path
from PyPDF2 import PdfReader  # Using PyPDF2 for PDF reading


# Making a MockAPI (to not use OpenAI API Request everytime)
class MockOpenAI(LLM):
    @property
    def _llm_type(self) -> str:
        return "OpenAI"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return f"Response for the prompt: {prompt}"


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


def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    return text


def str_to_document(text: str):
    page_content_part, metadata_part = text.split(" metadata=")

    page_content = page_content_part.split("page_content", 1)[1].strip("'")

    metadata = ast.literal_eval(metadata_part)

    return Document(page_content=page_content, metadata=metadata)


class Quanta:
    def __init__(self, llm=None):
        """Initialize the Quanta with a language model."""
        self.llm = llm or OpenAI()
        # self.llm = MockOpenAI()
        # Checking current model of llm - print(self.llm)
        self.document_store = self.get_doc_store()
        self.retriever = self.document_store.as_retriever()

    def get_doc_store(self):
        """Initialize ChromaDB as the document store for retrieval."""
        return Chroma(
            persist_directory="chroma_db", embedding_function=OpenAIEmbeddings()
        )

    def chunk_documents(self, file):
        """Chunk documents for further processing (summarization)."""
        """Load and write the documents into the ChromaDB document store."""
        if file.name.endswith(".docx"):
            loader = TextLoader(file.name)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file.name)
        elif file.name.endswith(".pdf"):
            loader = CustomPDFLoader(file)

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=0
        )

        raw_chunks = text_splitter.split_documents(documents)

        chunks = [str(documents) for documents in raw_chunks]
        chunks = [preprocess_text(chunk) for chunk in chunks]
        docs = [str_to_document(chunk) for chunk in chunks]

        print(len(docs))

        return docs

    def query_chain(self, query):
        """Run a query using the LangChain-based QA chain with ChromaDB."""
        qa_chain = retrieval_qa(llm=self.llm, retriever=self.retriever)
        result = qa_chain.run(query)
        return result

    # -------------------START SUMMARY-----------------------#
    def summary_tool(self, file):
        """Summarize an entire document by splitting it into chunks, summarizing, and reducing."""
        # Step 1: Chunk the original document
        chunked_docs = self.chunk_documents(file)

        # Step 2: Summarize each chunk
        summaries = [self.map_summarizer(self, chunk) for chunk in chunked_docs]

        batch_size = 10
        combined_summaries = ""

        # Loop over chunked_docs in batches of size 10
        for i in range(0, len(summaries), batch_size):
            batch = summaries[
                i : i + batch_size
            ]  # Get a batch of 10 (or fewer if at the end)

            combined_text = " ".join([doc for doc in batch])

            # Create a new LangChain Document with the combined text
            combined_document = Document(page_content=combined_text)

            combined_summaries += self.map_summarizer(self, combined_document)

        # Step 3: Reduce multiple summaries into a final summary
        final_summary = self.reduce_summarizer(combined_summaries)

        return final_summary

    # Summarize into multiple chunk of text based on user's query.
    def map_summarizer(self, query, chunk):
        prompt_template = PromptTemplate(
            input_variables=["query", "chunk"],
            template="""
            Keep in mind that this is most likely a research paper.
            You are responsible for summarizing a chunk of text according to a user’s query.
            User query: {query}
            Text chunk to summarize: {chunk}
            Provide ONE to TWO SENTENCE summary based on the user’s query.
            """,
        )
        prompt = prompt_template.format(query=query, chunk=chunk)
        summary = self.llm(prompt)
        return summary

    # With the multiple chunks, further summarize them by combining those different chunks.
    def reduce_summarizer(self, summaries):
        combined_summaries = " ".join(summaries)
        prompt_template = PromptTemplate(
            input_variables=["summaries"],
            template="""
            You are responsible for combining multiple summaries into couple paragraphs.
            Here are the summaries: {summaries}
            Provide an ACCURATE summary. DOUBLE CHECK the GRAMMAR.
            """,
        )
        prompt = prompt_template.format(summaries=combined_summaries)
        reduced_summary = self.llm(prompt)
        return reduced_summary

    # -------------------END SUMMARY-----------------------#

    # For free talking.
    def simple_responder(self, query):
        simple_prompt = (
            f"Please generate a simple response to the following query: '{query}'"
        )
        response = self.llm(simple_prompt)
        return response

    # def context_tool(self, query):
    #     """Retrieve relevant context for a query from the document store."""
    #     retriever = self.document_store.as_retriever()
    #     context_docs = retriever.get_relevant_documents(query)

    #     # Format the documents into a concise context
    #     prompt_template = PromptTemplate(
    #         input_variables=["docs"],
    #         template="""
    #         You are a professional assistant for a chatbot system.
    #         Based on the following documents, provide a relevant context for the user's query.
    #         Documents: {docs}
    #         """
    #     )
    #     docs_content = " ".join([doc.page_content for doc in context_docs])
    #     prompt = prompt_template.format(docs=docs_content)

    #     context_response = self.llm(prompt)
    #     return context_response


# --------------START Streamlit Interface--------------------#
def streamlit_ui():
    st.title("Quanta-Bot for Researchers.")

    # Use mock for testing purposes
    # Initialize the quanta (it will use the mock OpenAI API)
    quanta = Quanta()

    # File uploader
    file = st.file_uploader(
        "Upload your document (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"]
    )

    # Query input
    query = st.text_input("Enter your query to Quanta!")

    # User query (input) for requesting particular chatbot service.
    if query:
        st.write("Processing...")
        if file and (
            "summarize" in query.lower()
            or "summary" in query.lower()
            or "summaries" in query.lower()
        ):
            summary = quanta.summary_tool(file)
            st.write("Here is your summary:")
            st.write(summary)
        # --> Other functionalities goes here.
        else:  # any other features can be added in the future
            retreived_info = quanta.simple_responder(file)
            st.write(retreived_info)


if __name__ == "__main__":
    streamlit_ui()
# --------------END Streamlit Interface--------------------#


# TODO: continuous talking (back-and-forth)
# TODO: Uploading multiple PDFs.

# TODO : Add more error handling and logging
# TODO : Add more comments and docstrings
# TODO : Cut down on the time it takes for response
# TODO : Add streaming responses
# TODO : Add self rag loop through invoking quanta until answer is found
# TODO : Add URL abilities
# TODO : Conversation History: The quanta can access conversation history to maintain context and provide more relevant responses.
# TODO : Document Summarization: It can summarize documents to provide concise answers or overviews.
# TODO : Follow-up Answers: The quanta can answer follow-up questions based on previous interactions and the current conversation context.
# TODO : Logical Intent Determination: It uses logic to determine user intent, ensuring accurate responses.
# TODO : UI Streamlit 건들여서 발전 시켜보기
