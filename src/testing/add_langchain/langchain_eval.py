import os
import time
import tracemalloc
import pandas as pd
from tqdm import tqdm
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sklearn.metrics import ndcg_score, f1_score

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import FakeListLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.language_models.llms import BaseLLM
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

# 1. Load BEIR dataset
os.environ["OMP_NUM_THREADS"] = "1"
dataset = "nfcorpus"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# 2. Subset for balanced testing - medium sample size
LIMIT_DOCS = 500
LIMIT_QUERIES = 50
doc_ids = list(corpus.keys())[:LIMIT_DOCS]
documents = []
for doc_id in doc_ids:
    # Create document objects with metadata for parent retrieval
    from langchain_core.documents import Document
    doc = Document(
        page_content=corpus[doc_id]['text'],
        metadata={"source": doc_id}
    )
    documents.append(doc)

query_ids = list(queries.keys())[:LIMIT_QUERIES]

# 3. Embedding + Parent Document Retrieval setup
model_name = "all-mpnet-base-v2"
embed_model = HuggingFaceEmbeddings(model_name=model_name)

# Create empty vector store for parent document retriever
vectorstore = FAISS.from_texts(["temp"], embed_model)
vectorstore.delete([vectorstore.index_to_docstore_id[0]])  # Remove temp document

# Set up parent document retrieval with hierarchical chunking
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Parent chunks
child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)   # Child chunks for search

# Create in-memory store for parent documents
store = InMemoryStore()

# 4. Custom Fake LLM for Multi-Query that generates query variations
from langchain_core.outputs import LLMResult, Generation
from typing import List, Optional, Any

class SimpleQueryGeneratorLLM(BaseLLM):
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> LLMResult:
        generations = []
        for prompt in prompts:
            # Simple query variation generation
            if "Original question:" in prompt:
                original_query = prompt.split("Original question:")[1].strip().split("\n")[0]
            else:
                # Extract query from different prompt formats
                lines = prompt.split("\n")
                original_query = lines[-1] if lines else "sample query"

            # Generate simple variations for multi-query
            variations = [
                f"{original_query}",
                f"What is {original_query}?",
                f"Information about {original_query}"
            ]
            result_text = "\n".join(variations)
            generations.append([Generation(text=result_text)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self):
        return "simple_query_generator"

# LLMs for different purposes
query_gen_llm = SimpleQueryGeneratorLLM()
response_llm = FakeListLLM(responses=["This is a fake answer."] * 1000)

prompt = PromptTemplate.from_template("Answer the question using the context:\n\n{context}\n\nQuestion: {input}")
combine_docs_chain = create_stuff_documents_chain(llm=response_llm, prompt=prompt)

# Create Parent Document Retriever with hierarchical chunking
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    k=5,  # Return top 5 child chunks
)

# Add documents to parent retriever
parent_retriever.add_documents(documents, ids=None)

# Create prompt for multi-query generation
multiquery_prompt = PromptTemplate.from_template(
    "You are an AI language model assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database.\n\nOriginal question: {question}\n\nProvide these alternative questions separated by newlines:"
)

# First wrap Parent Document Retriever with Multi-Query for query expansion
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=parent_retriever,
    llm=query_gen_llm,
    prompt=multiquery_prompt
)

# Then add Cross-Encoder Reranker for better ranking
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(
    model=cross_encoder,
    top_n=8  # Rerank top 8 results
)

# Final retriever with all three features: Parent-Document + Multi-Query + Reranking
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=multi_query_retriever
)

retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# 5. Metric function
def compute_metrics(retrieved, ground_truth, doc_id_map):
    relevance = [1 if doc_id in ground_truth else 0 for doc_id in retrieved]
    ideal_relevance = sorted(relevance, reverse=True)
    ndcg = ndcg_score([ideal_relevance], [relevance])
    y_true = [1 if doc_id in ground_truth else 0 for doc_id in doc_id_map]
    y_pred = [1 if doc_id in retrieved else 0 for doc_id in doc_id_map]
    f1 = f1_score(y_true, y_pred, average='macro')
    mrr = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in ground_truth:
            mrr = 1 / i
            break
    recall_at_5 = len(set(retrieved[:5]) & ground_truth) / len(ground_truth) if ground_truth else 0.0
    return ndcg, f1, mrr, recall_at_5

# 6. Run and collect results
results = []
for query_id in tqdm(query_ids):
    query = queries[query_id]
    ground_truth = set(qrels[query_id].keys()).intersection(set(doc_ids))
    if not ground_truth:
        continue

    tracemalloc.start()
    t0 = time.time()
    result = retrieval_chain.invoke({"input": query})
    t1 = time.time()
    mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Extract retrieved docs
    sources = result["context"]
    retrieved_ids = []
    for doc in sources:
        # Use metadata source if available, otherwise try to match content
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            retrieved_ids.append(doc.metadata['source'])
        else:
            # Fallback: try to match content to original doc_ids
            for doc_id in doc_ids:
                if doc.page_content.strip() in corpus[doc_id]['text'] or corpus[doc_id]['text'] in doc.page_content.strip():
                    retrieved_ids.append(doc_id)
                    break

    ndcg, f1, mrr, recall = compute_metrics(retrieved_ids, ground_truth, doc_ids)
    results.append({
        "query_id": query_id,
        "method": "LangChain+MPNet",
        "NDCG": ndcg,
        "F1": f1,
        "MRR": mrr,
        "Recall@5": recall,
        "Time_ms": (t1 - t0) * 1000,
        "Memory_bytes": mem
    })

# 7. Save results
df = pd.DataFrame(results)
df.to_csv("langchain_mpnet_results.csv", index=False)
print("Saved: langchain_mpnet_results.csv")
