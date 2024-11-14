from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from ..utils.document_processor import DocumentProcessor
from ..utils.token_counter import estimate_tokens, calculate_optimal_batch

# Main component of the Chatbot Quanta.
class Quanta:
    def __init__(self, llm=None):
        """Initialize the Quanta with a language model."""
        self.llm = llm or OpenAI()
        self.document_store = self.get_doc_store()
        self.retriever = self.document_store.as_retriever()

    def get_doc_store(self):
        # Initialize ChromaDB as the document store for retrieval.
        return Chroma(
            persist_directory="chroma_db",
            embedding_function=OpenAIEmbeddings()
        )

    def chunk_documents(self, files, sentences_per_chunk=5):
        return DocumentProcessor.chunk_documents(files, self.document_store)

    def query_chain(self, query):
        if not self.retriever:
            return "Please upload documents first before asking questions."
            
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        
        result = qa_chain({"query": query})
        return result["result"]

    def summary_tool(self, file):
        # Step 1: Chunk the original document
        chunked_docs = self.chunk_documents(file)
        
        chunk_summaries = []
        for chunk in chunked_docs:
            summary = self.map_summarizer(self, chunk)
            chunk_summaries.append(summary)

        # Calculate optimal batch size for these summaries
        batch_size = calculate_optimal_batch(chunk_summaries, estimate_tokens)
        
        # Step 2: Summarize each chunk
        summaries = [self.map_summarizer(self, chunk) for chunk in chunked_docs]
        combined_summaries = ""

        # Through summaries, we now have each chunks into one sentence summary.
        # Now, we are adding multiple chunks (with 1 sentence) to go through map_summarizer ONCE MORE,
        # to ensure that when contents are sent to map_summarizer, it isn't too long.
        # Loop over chunked_docs in specific batch size.
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i : i + batch_size]
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
            You are responsible for summarizing a chunk of text according to a user's query.
            User query: {query}
            Text chunk to summarize: {chunk}
            Provide ONE SENTENCE summary based on the user's query.
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
            Provide an ACCURATE summary. DOUBLE CHECK the GRAMMAR. Ensure sentences are COMPLETE.
            """,
        )
        prompt = prompt_template.format(summaries=combined_summaries)
        reduced_summary = self.llm(prompt)
        return reduced_summary

    def simple_responder(self, query):
        simple_prompt = f"Please generate a simple response to the following query: '{query}'"
        response = self.llm(simple_prompt)
        return response
