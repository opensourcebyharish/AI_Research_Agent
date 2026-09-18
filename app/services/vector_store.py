import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")

        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        self.metadata = []

    def add(self, embeddings, chunks, metadata):
        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks"
            )

        if len(chunks) != len(metadata):
            raise ValueError(
                "Number of chunks must match number of metadata entries"
            )

        if not embeddings:
            return

        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embeddings must have shape (n, {self.dimension})"
            )

        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k=5):
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding must have dimension {self.dimension}"
            )

        k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(
            query_vector,
            k,
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0],
        ):
            results.append(
                {
                    "chunk": self.chunks[index],
                    "distance": float(distance),
                    "metadata": self.metadata[index],
                }
            )

        return results