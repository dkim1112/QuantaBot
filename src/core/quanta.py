from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from ..utils.document_processor import DocumentProcessor
from ..utils.token_counter import estimate_tokens, calculate_optimal_batch

# Main component of the Chatbot Quanta.
class Quanta:
    def __init__(self, llm=None):
        """Initialize the Quanta with an OpenAI model."""
        self.llm = llm or OpenAI()
        self.document_store = self.get_doc_store()
        self.retriever = self.document_store.as_retriever()

    def get_doc_store(self):
        # Initialize ChromaDB as the document store for retrieval.
        return Chroma(
            persist_directory="chroma_db",
            embedding_function=OpenAIEmbeddings()
        )
    
    def reset_document_store(self):
        # # Clear the persistent directory if it exists
        # persist_dir = "chroma_db"
        # shutil.rmtree(persist_dir, ignore_errors=True)
        """Clear the document store to remove any previously uploaded documents."""
        self.document_store = self.get_doc_store()  # Reinitialize the store
        self.retriever = self.document_store.as_retriever()

    def chunk_documents(self, files, sentences_per_chunk=5):
        self.reset_document_store()
        chunked_docs = DocumentProcessor.chunk_documents(files, self.document_store)
        return chunked_docs

    def query_chain(self, files, query):
        chunked_docs = self.chunk_documents(files)
        
        chunk_contents = [chunk.page_content for chunk in chunked_docs]

        batch_size = calculate_optimal_batch(chunk_contents, estimate_tokens)

        final_responses = []
    
        # Aggregate responses from batches
        for i in range(0, len(chunked_docs), batch_size):
            # Select a batch of chunks
            batch = chunk_contents[i:i+batch_size]
            
            # Combine batch chunks into a single context
            batch_context = " ".join(batch)
            
            # Create a prompt that includes the query and batch context
            batch_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="""
                Given the following context from a document and the query, 
                provide a focused response for the QUERY specifically:
                
                Query: {query}
                Context: {context}
                
                Response:
                """,
            )
            
            # Format the prompt
            formatted_prompt = batch_prompt.format(query=query, context=batch_context)
            
            # Get response for this batch
            batch_response = self.llm(formatted_prompt)
            
            # Append batch responses
            final_responses.append(batch_response)
            
        return self.reduce_queries(final_responses)

    def summary_tool(self, file):
        # Step 1: Chunk the original document, and save each chunks into one unique index of the chunked_docs list.
        chunked_docs = self.chunk_documents(file)
        
        # Step 2: Iterating through the chunked_docs list above, we summarize each index and added to a NEW list,
        # called the chunked_summaries.
        chunk_summaries = []
        for chunk in chunked_docs:
            summary = self.map_summarizer(self, chunk)
            chunk_summaries.append(summary)

        # Step 3: Depending on the size of the chunk_summaries, batch size should be determined
        # before sending to LLM. This is to prevent token-limit error.
        # Calculate optimal batch size for these summaries.
        batch_size = calculate_optimal_batch(chunk_summaries, estimate_tokens)
        
        # Step 4: Summarize each batches OF chunks and save it here as a string.
        combined_summaries = ""

        # Through summaries above, we now have each chunks into one sentence summary.
        # Now, we are adding multiple chunks (with 1 sentence) = ONE batch each, to go through map_summarizer ONCE MORE.
        for i in range(0, len(chunk_summaries), batch_size):
            batch = chunk_summaries[i : i + batch_size]
            combined_text = " ".join([doc for doc in batch])

            # Create a new LangChain Document with the combined text
            combined_document = Document(page_content=combined_text)
            combined_summaries += self.map_summarizer(self, combined_document)
            # As a result, combined_summaries is a STRING full of necessary sentences. Might be quite long.

        # Step 5: Final summarize the combined_summaries.
        final_summary = self.reduce_summarizer(combined_summaries)
        return final_summary
        
    # Summarize text(s) based on user's query. Will return one sentence summary every call.
    # This is NOT the final summary.
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

    # Determinant for FINAL Summary output.
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

    def reduce_queries(self, final_response):
            final_response = " ".join(final_response)
            prompt_template = PromptTemplate(
                input_variables=["final_response"],
                template="""
                You are responsible for combining multiple sentences into paragraphs or sentences, depending on the length.
                Here are the responses: {final_response}
                Within all those texts, there's probably a commonality of the answers. 
                Explain those answers so that it provides a clear answer to the question, without mentioning that they are commonalities and those were the answers/outputs given to you.
                """,
            )
            prompt = prompt_template.format(final_response=final_response)
            reduced_response = self.llm(prompt)
            return reduced_response
    
    # NOT DONE YET
    def simple_responder(self, query):
        simple_prompt = f"Please generate a simple response to the following query: '{query}'"
        response = self.llm(simple_prompt)
        return response
