import os
import time
import faiss
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, f1_score
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load dataset
os.environ["OMP_NUM_THREADS"] = "1"
dataset = "nfcorpus"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Limit documents and queries for performance
LIMIT_DOCS = 2000
LIMIT_QUERIES = 100
doc_id_map = list(corpus.keys())[:LIMIT_DOCS]
texts = [corpus[doc_id]['text'] for doc_id in doc_id_map]
query_ids = list(queries.keys())[:LIMIT_QUERIES]

# Calculate document lengths for scaling (word count)
doc_lengths = [len(text.split()) for text in texts]

# Models to test
models_to_test = {
    "MiniLM": "all-MiniLM-L6-v2",
    "MPNet": "all-mpnet-base-v2",
    "DistilUSE": "distiluse-base-multilingual-cased",
    "NLI-BERT": "bert-base-nli-mean-tokens",
    "SciBERT": "allenai-specter",  # Note: specter is based on SciBERT and optimized for scientific docs
}

# Similarity strategies
index_types = {
    "L2": lambda dim: faiss.IndexFlatL2(dim), # (Euclidean - most commonly used)
    "Cosine": lambda dim: faiss.IndexFlatIP(dim)
}

# Normalization function for cosine similarity
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, a_min=1e-10, a_max=None)

# Metric functions
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

results = []

for model_name, model_path in models_to_test.items():
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_path)
    base_embeddings = model.encode(texts, convert_to_numpy=True)

    for similarity_name, index_func in index_types.items():
        print(f"\nTesting {model_name} with {similarity_name}")
        if similarity_name == "Cosine":
            embeddings = normalize(base_embeddings) # Normalization needs to be used for Cos. Similarity
        else:
            # For L2: Scale embeddings by document length to show magnitude effects
            # Longer documents get higher magnitude (more "important")
            length_scaling = np.array(doc_lengths).reshape(-1, 1) / np.mean(doc_lengths)
            embeddings = base_embeddings * length_scaling

        index = index_func(embeddings.shape[1])
        index.add(embeddings)

        query_scores = {"NDCG": [], "F1": [], "MRR": [], "Recall@5": [], "Memory_bytes": []}

        for query_id in tqdm(query_ids):
            query = queries[query_id]
            ground_truth = set(qrels[query_id].keys()).intersection(set(doc_id_map))
            if not ground_truth:
                continue

            query_embedding = model.encode([query], convert_to_numpy=True)
            if similarity_name == "Cosine":
                query_embedding = normalize(query_embedding)
            else:
                # For L2: Scale query by its word count (like documents)
                query_length = len(query.split())
                query_scaling = query_length / np.mean(doc_lengths)
                query_embedding = query_embedding * query_scaling

            tracemalloc.start() # Start memory tracking
            _, indices = index.search(query_embedding, 5)
            mem = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

            retrieved = [doc_id_map[i] for i in indices[0]]
            ndcg, f1, mrr, recall = compute_metrics(retrieved, ground_truth, doc_id_map)
            query_scores["NDCG"].append(ndcg)
            query_scores["F1"].append(f1)
            query_scores["MRR"].append(mrr)
            query_scores["Recall@5"].append(recall)
            query_scores["Memory_bytes"].append(mem)

        result = {
            "model": model_name,
            "similarity": similarity_name,
            "NDCG_mean": np.mean(query_scores["NDCG"]),
            "F1_mean": np.mean(query_scores["F1"]),
            "MRR_mean": np.mean(query_scores["MRR"]),
            "Recall@5_mean": np.mean(query_scores["Recall@5"]),
            "Memory_bytes_mean": np.mean(query_scores["Memory_bytes"]),
            "NDCG_std": np.std(query_scores["NDCG"]),
            "F1_std": np.std(query_scores["F1"]),
            "MRR_std": np.std(query_scores["MRR"]),
            "Recall@5_std": np.std(query_scores["Recall@5"]),
            "Memory_bytes_std": np.std(query_scores["Memory_bytes"])
        }
        results.append(result)

# Save results
df = pd.DataFrame(results)
df.to_csv("embeddings_all_metrics.csv", index=False)
print("Saved: embeddings_all_metrics.csv")
