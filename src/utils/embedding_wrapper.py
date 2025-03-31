from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        vectors = self.model.encode(texts, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors] # List[List[float]]

    def embed_query(self, text):
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist() # List[float]
