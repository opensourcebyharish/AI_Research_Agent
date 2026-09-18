class RetrievalService:
    """Retrieve relevant document chunks for a user query."""

    def __init__(
        self,
        embedding_service,
        vector_store,
        max_distance: float | None = 1.5,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.max_distance = max_distance

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ):
        """
        Retrieve relevant chunks using semantic similarity.

        FAISS uses L2 distance, so smaller distances indicate
        more similar vectors.

        Args:
            query: User's question.
            top_k: Maximum number of chunks to retrieve.

        Returns:
            Relevant retrieval results.
        """

        if not query or not query.strip():
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        query_embedding = self.embedding_service.embed_text(
            query
        )

        if not query_embedding:
            return []

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        if self.max_distance is None:
            return results

        return [
            result
            for result in results
            if result["distance"] <= self.max_distance
        ]