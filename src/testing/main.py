import os
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent FAISS from using multiple threads
import json
import faiss
import numpy as np
import pandas as pd
import chromadb
import time
import tracemalloc
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score, f1_score
from sentence_transformers import SentenceTransformer

# Load BEIR dataset
dataset = "nfcorpus"  # Change to desired dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Convert corpus into list of documents and maintain mapping
doc_id_map = list(corpus.keys())  # Keep original string-based document IDs
texts = [corpus[doc_id]['text'] for doc_id in doc_id_map]
query_ids = list(queries.keys())
query = queries[query_ids[0]]  # Select a sample query

ground_truth = set(qrels[query_ids[0]].keys())  # Keep as strings

def compute_ndcg(retrieved, ground_truth):
    relevance = [1 if doc_id in ground_truth else 0 for doc_id in retrieved]
    ideal_relevance = sorted(relevance, reverse=True)
    return ndcg_score([ideal_relevance], [relevance])

def compute_f1(retrieved, ground_truth):
    y_true = [1 if doc_id in ground_truth else 0 for doc_id in doc_id_map]
    y_pred = [1 if doc_id in retrieved else 0 for doc_id in doc_id_map]
    return f1_score(y_true, y_pred, average='macro')

def compute_mrr(retrieved, ground_truth):
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in ground_truth:
            return 1 / i  # Reciprocal rank of the first relevant document
    return 0.0

def compute_recall_at_k(retrieved, ground_truth, k=5):
    retrieved_set = set(retrieved[:k])
    relevant_set = ground_truth
    return len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0.0

##### TF-IDF + COSINE SIMILARITY + FAISS #####
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(texts).toarray()
query_vector = vectorizer.transform([query]).toarray()

tfidf_index = faiss.IndexFlatL2(doc_vectors.shape[1])
tfidf_index.add(doc_vectors)

tracemalloc.start()
t0 = time.time()
distances, indices = tfidf_index.search(query_vector, 5)
t1 = time.time()
mem_tfidf = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()

# Cleanup FAISS indices after use


import gc
gc.collect()  # Run garbage collection only once at the end

# Ensure all FAISS indices are cleaned up after each retrieval method


gc.collect()  # Run garbage collection only once at the end


tfidf_results = [doc_id_map[i] for i in indices[0]]
tfidf_ndcg = compute_ndcg(tfidf_results, ground_truth)
tfidf_f1 = compute_f1(tfidf_results, ground_truth)
tfidf_mrr = compute_mrr(tfidf_results, ground_truth)
tfidf_recall = compute_recall_at_k(tfidf_results, ground_truth)
print("TF-IDF + FAISS Results:", tfidf_results)
print(f"TF-IDF Speed: {round((t1 - t0) * 1000, 2)} ms, NDCG: {tfidf_ndcg}, F1-score: {tfidf_f1}, MRR: {tfidf_mrr}, Recall@5: {tfidf_recall}, Memory: {mem_tfidf} bytes")

##### BERT EMBEDDINGS + FAISS #####
from transformers import BertModel, BertTokenizer
import torch

bert_model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
embeddings = np.array([get_bert_embedding(text) for text in texts])
query_embedding = np.array(get_bert_embedding(query))

faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

tracemalloc.start()
t0 = time.time()
distances, indices = faiss_index.search(query_embedding, 5)
t1 = time.time()
mem_faiss = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()

faiss_results = [doc_id_map[i] for i in indices[0]]
faiss_ndcg = compute_ndcg(faiss_results, ground_truth)
faiss_f1 = compute_f1(faiss_results, ground_truth)
faiss_mrr = compute_mrr(faiss_results, ground_truth)
faiss_recall = compute_recall_at_k(faiss_results, ground_truth)
print("BERT + FAISS Results:", faiss_results)
print(f"FAISS Speed: {round((t1 - t0) * 1000, 2)} ms, NDCG: {faiss_ndcg}, F1-score: {faiss_f1}, MRR: {faiss_mrr}, Recall@5: {faiss_recall}, Memory: {mem_faiss} bytes")

##### BERT EMBEDDINGS + COSINE SIMILARITY + FAISS #####
tracemalloc.start()
t0 = time.time()
cosine_sim_scores = cosine_similarity(query_embedding, embeddings).flatten()
cosine_results = [doc_id_map[i] for i in np.argsort(cosine_sim_scores)[::-1][:5]]
t1 = time.time()
mem_cosine = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()

cosine_ndcg = compute_ndcg(cosine_results, ground_truth)
cosine_f1 = compute_f1(cosine_results, ground_truth)
cosine_mrr = compute_mrr(cosine_results, ground_truth)
cosine_recall = compute_recall_at_k(cosine_results, ground_truth)
print("BERT + Cosine Similarity + FAISS Results:", cosine_results)
print(f"Cosine Similarity Speed: {round((t1 - t0) * 1000, 2)} ms, NDCG: {cosine_ndcg}, F1-score: {cosine_f1}, MRR: {cosine_mrr}, Recall@5: {cosine_recall}, Memory: {mem_cosine} bytes")

##### BERT EMBEDDINGS + PCA + COSINE SIMILARITY + FAISS #####
pca = PCA(n_components=100)
pca_embeddings = pca.fit_transform(embeddings)
pca_query_embedding = pca.transform(query_embedding)

tracemalloc.start()
t0 = time.time()
pca_sim_scores = cosine_similarity(pca_query_embedding, pca_embeddings).flatten()
pca_results = [doc_id_map[i] for i in np.argsort(pca_sim_scores)[::-1][:5]]
t1 = time.time()
mem_pca = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()

pca_ndcg = compute_ndcg(pca_results, ground_truth)
pca_f1 = compute_f1(pca_results, ground_truth)
pca_mrr = compute_mrr(pca_results, ground_truth)
pca_recall = compute_recall_at_k(pca_results, ground_truth)
print("BERT + PCA + Cosine Similarity + FAISS Results:", pca_results)
print(f"PCA Cosine Similarity Speed: {round((t1 - t0) * 1000, 2)} ms, NDCG: {pca_ndcg}, F1-score: {pca_f1}, MRR: {pca_mrr}, Recall@5: {pca_recall}, Memory: {mem_pca} bytes")
