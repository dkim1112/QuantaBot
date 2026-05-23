import os

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"


class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name=None):
        # Resolution order: explicit arg → QUANTA_EMBEDDING_MODEL env var → mpnet default.
        # Set QUANTA_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5 for the BGE upgrade.
        model_name = model_name or os.getenv("QUANTA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors] # List[List[float]]

    def embed_query(self, text):
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist() # List[float]
