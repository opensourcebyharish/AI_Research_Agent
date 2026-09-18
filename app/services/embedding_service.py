from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        if not text or not text.strip():
            return []

        return self.model.encode(text).tolist()

    def embed_chunks(self, chunks: list[str]):
        if not chunks:
            return []

        return self.model.encode(chunks).tolist()