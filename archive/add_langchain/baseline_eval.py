import os
import time
import tracemalloc
import pandas as pd
from tqdm import tqdm
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# Settings
os.environ["OMP_NUM_THREADS"] = "1"
dataset = "nfcorpus"
LIMIT_DOCS = 500
LIMIT_QUERIES = 50

# Load BEIR dataset
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Add fake documents to corpus for testing
# This helps testing out if LangChain is really beneficial.
num_distractors = int(LIMIT_DOCS * 0.2)
for i in range(num_distractors):
    fake_id = f"fake-{i}"
    corpus[fake_id] = {"text": "This document is about unrelated topics like cats, weather, and gardening."}

doc_ids = list(corpus.keys())[:LIMIT_DOCS + num_distractors]
texts = [corpus[doc_id]['text'] for doc_id in doc_ids]
query_ids = list(queries.keys())[:LIMIT_QUERIES]

# MPNet embeddings (best performing model from comparison)
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

def get_embedding(text):
    return model.encode(text, convert_to_numpy=True)

doc_embeddings = [get_embedding(text) for text in texts]

results = []

for query_id in tqdm(query_ids):
    query = queries[query_id]
    ground_truth = set(qrels[query_id].keys()).intersection(set(doc_ids))
    if not ground_truth:
        continue

    query_embedding = get_embedding(query)

    tracemalloc.start()
    t0 = time.time()
    sims = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_k = sims.argsort()[::-1][:15] # Get top 15 results
    t1 = time.time()
    mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    retrieved = [doc_ids[i] for i in top_k]

    relevance = [1 if doc_id in ground_truth else 0 for doc_id in retrieved]
    ideal_relevance = sorted(relevance, reverse=True)
    ndcg = ndcg_score([ideal_relevance], [relevance])

    y_true = [1 if doc_id in ground_truth else 0 for doc_id in doc_ids]
    y_pred = [1 if doc_id in retrieved else 0 for doc_id in doc_ids]
    f1 = f1_score(y_true, y_pred, average='macro')

    mrr = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in ground_truth:
            mrr = 1 / i
            break

    recall = len(set(retrieved[:5]) & ground_truth) / len(ground_truth) if ground_truth else 0.0

    results.append({
        "query_id": query_id,
        "method": "Baseline+MPNet",
        "NDCG": ndcg,
        "F1": f1,
        "MRR": mrr,
        "Recall@5": recall,
        "Time_ms": (t1 - t0) * 1000,
        "Memory_bytes": mem
    })

# Save results
df = pd.DataFrame(results)
df.to_csv("baseline_mpnet_results.csv", index=False)
print("Saved: baseline_mpnet_results.csv")
