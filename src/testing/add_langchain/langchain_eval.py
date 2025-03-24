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

# 1. Load BEIR dataset
os.environ["OMP_NUM_THREADS"] = "1"
dataset = "nfcorpus"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# 2. Subset for quick testing
LIMIT_DOCS = 500
LIMIT_QUERIES = 50
doc_ids = list(corpus.keys())[:LIMIT_DOCS]
texts = [corpus[doc_id]['text'] for doc_id in doc_ids]
query_ids = list(queries.keys())[:LIMIT_QUERIES]

# 3. Embedding + Vector Store setup
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbeddings(model_name=model_name)
vectorstore = FAISS.from_texts(texts, embed_model)

# 4. Fake LLM and retrieval chain setup
llm = FakeListLLM(responses=["This is a fake answer."] * 1000)
prompt = PromptTemplate.from_template("Answer the question using the context:\n\n{context}\n\nQuestion: {input}")
combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
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
        try:
            idx = texts.index(doc.page_content)
            retrieved_ids.append(doc_ids[idx])
        except ValueError:
            continue

    ndcg, f1, mrr, recall = compute_metrics(retrieved_ids, ground_truth, doc_ids)
    results.append({
        "query_id": query_id,
        "method": "LangChain+Vector",
        "NDCG": ndcg,
        "F1": f1,
        "MRR": mrr,
        "Recall@5": recall,
        "Time_ms": (t1 - t0) * 1000,
        "Memory_bytes": mem
    })

# 7. Save results
df = pd.DataFrame(results)
df.to_csv("langchain_vector_retrieval_results.csv", index=False)
print("Saved: langchain_vector_retrieval_results.csv")
