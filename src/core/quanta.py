import pickle
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from ..utils.document_processor import DocumentProcessor
from ..utils.token_counter import estimate_tokens, calculate_optimal_batch # NO LONGER IN USE?

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

    def save_embeddings_for_projector(self, embeddings, documents, file_prefix="embeddings"):
        """Save embeddings and metadata in TSV format for TensorFlow Projector."""
        embeddings_array = np.array(embeddings)

        # Save embeddings as .tsv
        embeddings_path = f"{file_prefix}_vectors.tsv"
        np.savetxt(embeddings_path, embeddings_array, delimiter="\t")
        print(f"📂 Embeddings saved: {embeddings_path}")

        # Print sample text previews before saving metadata
        print("🔍 Sample text previews for metadata:")
        for i, doc in enumerate(documents[:5]):  # Print first 5 documents
            print(f"📄 Document {i}: {doc.page_content[:200]}")  # Show first 200 characters

        # Save metadata (document text preview)
        metadata_path = f"{file_prefix}_metadata.tsv"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("Document ID\tText Preview\n")  # Header
            for i, doc in enumerate(documents):
                preview = doc.page_content[:100].replace("\n", " ")  # First 100 chars for better readability
                preview = ''.join(c if c.isprintable() else '?' for c in preview)  # Replace unprintable chars
                if len(preview.strip()) == 0:
                    preview = "[EMPTY CHUNK]"  # Mark empty chunks to identify issues
                f.write(f"{i}\t{preview}\n")
        print(f"📂 Metadata saved: {metadata_path}")

    def preprocess_pipeline(self, files, n_components=50, n_clusters=10):
        # chunk the documents
        documents = DocumentProcessor.chunk_documents(files, self.document_store)

        # generate document embeddings
        embeddings = [OpenAIEmbeddings().embed_query(doc.page_content) for doc in documents]
        # print(embeddings)


        if len(embeddings) < n_components:
            print(f"Reducing n_components from {n_components} to {len(embeddings)} due to insufficient samples.")
            n_components = len(embeddings)
        # apply PCA for dimensionality reduction
        # parameter includes how many components? --> 50 (but can be changed up in the parameter)

        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        pca = PCA(n_components=n_components)

        # reduced_embeddings = pca.fit_transform(embeddings)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        reduced_embeddings = pca.fit_transform(embeddings)

        # wb = open in binary mode and write on it
        with open(os.path.join(model_dir, "pca_model.pkl"), "wb") as f:
            pickle.dump(pca, f)

        # cluster the embeddings based on kmeans
        # "random_state" = will have an effect on the reproducibility of the results returned by the function.
        n_clusters = min(n_clusters, len(reduced_embeddings))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(reduced_embeddings) # the one that has gone through pca

        with open(os.path.join(model_dir, "kmeans_model.pkl"), "wb") as f:
            pickle.dump(kmeans, f)
        
        # store the clustered data
        for idx, doc in enumerate(documents):
            embedding = reduced_embeddings[idx].tolist()
            doc.metadata["cluster"] = str(cluster_assignments[idx])
            doc.metadata["embedding"] = str(embedding)

        
        self.document_store.add_documents(documents)
        self.save_embeddings_for_projector(embeddings, documents)

    def query_pipeline(self, query, top_k_clusters=3, top_n_chunks=5):
        # generate query embeddings
        query_embedding = OpenAIEmbeddings().embed_query(query)
        # print(len(query_embedding))

        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        if query_embedding.size == 0:
            raise ValueError("Query embedding is empty. Ensure the input query is valid.")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # rb = open in binary mode and read
        model_dir = "models"
        pca_file = os.path.join(model_dir, "pca_model.pkl")
        kmeans_file = os.path.join(model_dir, "kmeans_model.pkl")

        with open(pca_file, "rb") as f:
            pca = pickle.load(f)
        with open(kmeans_file, "rb") as f:
            kmeans = pickle.load(f)


        # Step 3: Reduce Query Embedding Dimensionality - this is done as well b/c query embeddings needs to be reduced via PCA to match the reduced embeddings stored in document metadata.
        # reduced_query_embedding = pca.transform([query_embedding])[0]
        reduced_query_embedding = pca.transform(query_embedding)[0]

        # find relevant clusters using cosine similarity
        similarity = cosine_similarity(
            np.array([reduced_query_embedding]),  # Ensure 2D
            np.array(kmeans.cluster_centers_)    # Ensure 2D
        )[0]

        # argsort() = obtain the indices of the sorted elements
        relevant_clusters = similarity.argsort()[-top_k_clusters:][::-1]

        # retrieve candidate documents
        print(f"Retrieving documents for clusters: {relevant_clusters}")
        candidate_docs = []
        for cluster_id in relevant_clusters:
            cluster_docs = self.document_store.search(
                search_type="similarity",
                query="",
                filter={"cluster": str(cluster_id)},
                k=100,
            )
            print(f"Cluster {cluster_id} returned {len(cluster_docs)} documents.")
            candidate_docs.extend(cluster_docs)

        # perform a SEMANTIC SEARCH within clusters (which matches the most with documents uploaded)
        candidate_embeddings = []
        valid_docs = []
        for doc in candidate_docs:
            embedding = doc.metadata.get("embedding")
            if embedding:
                try:
                    parsed_embedding = eval(embedding)
                    if len(parsed_embedding) == len(reduced_query_embedding):
                        candidate_embeddings.append(parsed_embedding)
                        valid_docs.append(doc)
                    else:
                        # Normalize embedding dimensions (pad or truncate) - temporary solution
                        if len(parsed_embedding) < len(reduced_query_embedding):
                            padded_embedding = parsed_embedding + [0.0] * (len(reduced_query_embedding) - len(parsed_embedding))
                            candidate_embeddings.append(padded_embedding)
                        elif len(parsed_embedding) > len(reduced_query_embedding):
                            truncated_embedding = parsed_embedding[:len(reduced_query_embedding)]
                            candidate_embeddings.append(truncated_embedding)
                        valid_docs.append(doc)
                except Exception as e:
                    print(f"Failed to parse embedding for doc: {doc.metadata}, error: {e}")

        # Ensure valid_docs and candidate_embeddings are aligned
        candidate_docs = valid_docs

        # Check if there are any valid embeddings
        if not candidate_embeddings:
            print("No valid embeddings found after filtering.")
            return "No relevant documents were found for the query."

        # Convert to NumPy array for consistency
        candidate_embeddings = np.array(candidate_embeddings)
        print(f"Candidate embeddings shape: {candidate_embeddings.shape}")
        
        similarities = cosine_similarity([reduced_query_embedding], candidate_embeddings)[0]
        top_n_indices = similarities.argsort()[-top_n_chunks:][::-1]

        # Step 7: Retrieve Most Relevant Chunks
        relevant_chunks = [candidate_docs[idx].page_content for idx in top_n_indices]

        # Step 8: Generate Response Using PromptTemplate
        response = self.generate_response(query, relevant_chunks)
        return response


    def generate_response(self, query, chunks):
        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Keep in mind that this is most likely a research paper.
            You are a highly knowledgeable assistant. Answer the query below based on the provided context.
            User query: {query}
            Context: {context}
            Response:
            """,
        )
        context = "\n".join(chunks)
        prompt = prompt_template.format(query=query, context=context)
        return self.llm(prompt)
    

    # def reset_document_store(self):
    #     # # Clear the persistent directory if it exists
    #     # persist_dir = "chroma_db"
    #     # shutil.rmtree(persist_dir, ignore_errors=True)
    #     """Clear the document store to remove any previously uploaded documents."""
    #     self.document_store = self.get_doc_store()  # Reinitialize the store
    #     self.retriever = self.document_store.as_retriever()

    # def chunk_documents(self, files, sentences_per_chunk=5):
    #     self.reset_document_store()
    #     chunked_docs = DocumentProcessor.chunk_documents(files, self.document_store)
    #     return chunked_docs

    # def query_chain(self, files, query):
    #     chunked_docs = self.chunk_documents(files)
        
    #     chunk_contents = [chunk.page_content for chunk in chunked_docs]

    #     batch_size = calculate_optimal_batch(chunk_contents, estimate_tokens)

    #     final_responses = []
    
    #     # Aggregate responses from batches
    #     for i in range(0, len(chunked_docs), batch_size):
    #         # Select a batch of chunks
    #         batch = chunk_contents[i:i+batch_size]
            
    #         # Combine batch chunks into a single context
    #         batch_context = " ".join(batch)
            
    #         # Create a prompt that includes the query and batch context
    #         batch_prompt = PromptTemplate(
    #             input_variables=["query", "context"],
    #             template="""
    #             Given the following context from a document and the query, 
    #             provide a focused response for the QUERY specifically:
                
    #             Query: {query}
    #             Context: {context}
                
    #             Response:
    #             """,
    #         )
            
    #         # Format the prompt
    #         formatted_prompt = batch_prompt.format(query=query, context=batch_context)
            
    #         # Get response for this batch
    #         batch_response = self.llm(formatted_prompt)
            
    #         # Append batch responses
    #         final_responses.append(batch_response)
            
    #     return self.reduce_queries(final_responses)

    # def summary_tool(self, file):
    #     # Step 1: Chunk the original document, and save each chunks into one unique index of the chunked_docs list.
    #     chunked_docs = self.chunk_documents(file)
        
    #     # Step 2: Iterating through the chunked_docs list above, we summarize each index and added to a NEW list,
    #     # called the chunked_summaries.
    #     chunk_summaries = []
    #     for chunk in chunked_docs:
    #         summary = self.map_summarizer(self, chunk)
    #         chunk_summaries.append(summary)

    #     # Step 3: Depending on the size of the chunk_summaries, batch size should be determined
    #     # before sending to LLM. This is to prevent token-limit error.
    #     # Calculate optimal batch size for these summaries.
    #     batch_size = calculate_optimal_batch(chunk_summaries, estimate_tokens)
        
    #     # Step 4: Summarize each batches OF chunks and save it here as a string.
    #     combined_summaries = ""

    #     # Through summaries above, we now have each chunks into one sentence summary.
    #     # Now, we are adding multiple chunks (with 1 sentence) = ONE batch each, to go through map_summarizer ONCE MORE.
    #     for i in range(0, len(chunk_summaries), batch_size):
    #         batch = chunk_summaries[i : i + batch_size]
    #         combined_text = " ".join([doc for doc in batch])

    #         # Create a new LangChain Document with the combined text
    #         combined_document = Document(page_content=combined_text)
    #         combined_summaries += self.map_summarizer(self, combined_document)
    #         # As a result, combined_summaries is a STRING full of necessary sentences. Might be quite long.

    #     # Step 5: Final summarize the combined_summaries.
    #     final_summary = self.reduce_summarizer(combined_summaries)
    #     return final_summary
        
    # # Summarize text(s) based on user's query. Will return one sentence summary every call.
    # # This is NOT the final summary.
    # def map_summarizer(self, query, chunk):
    #     prompt_template = PromptTemplate(
    #         input_variables=["query", "chunk"],
    #         template="""
    #         Keep in mind that this is most likely a research paper.
    #         You are responsible for summarizing a chunk of text according to a user's query.
    #         User query: {query}
    #         Text chunk to summarize: {chunk}
    #         Provide ONE SENTENCE summary based on the user's query.
    #         """,
    #     )
    #     prompt = prompt_template.format(query=query, chunk=chunk)
    #     summary = self.llm(prompt)
    #     return summary

    # # Determinant for FINAL Summary output.
    # def reduce_summarizer(self, summaries):
    #     combined_summaries = " ".join(summaries)
    #     prompt_template = PromptTemplate(
    #         input_variables=["summaries"],
    #         template="""
    #         You are responsible for combining multiple summaries into couple paragraphs.
    #         Here are the summaries: {summaries}
    #         Provide an ACCURATE summary. DOUBLE CHECK the GRAMMAR. Ensure sentences are COMPLETE.
    #         """,
    #     )
    #     prompt = prompt_template.format(summaries=combined_summaries)
    #     reduced_summary = self.llm(prompt)
    #     return reduced_summary

    # def reduce_queries(self, final_response):
    #         final_response = " ".join(final_response)
    #         prompt_template = PromptTemplate(
    #             input_variables=["final_response"],
    #             template="""
    #             You are responsible for combining multiple sentences into paragraphs or sentences, depending on the length.
    #             Here are the responses: {final_response}
    #             Within all those texts, there's probably a commonality of the answers. 
    #             Explain those answers so that it provides a clear answer to the question, without mentioning that they are commonalities and those were the answers/outputs given to you.
    #             """,
    #         )
    #         prompt = prompt_template.format(final_response=final_response)
    #         reduced_response = self.llm(prompt)
    #         return reduced_response
    